"""
Script: run_conditional_fidelity.py
===================================
Round 6 (2026-11-08): conditional fidelity of certified reasons
against a NEURAL REFERENCE -- a multinomial linear head (logreg)
trained on the same predicted concept bits phi(x) the tree sees.

  conditional fidelity(R, c) = P( logreg(x) = c | phi(x) satisfies R )

R is sufficient for the tree's class c, so c is fixed by R; the number
measures how far the tree-certified reason transfers to the linear
model.  Grouped by reason PATTERN (images whose bits satisfy the
literal set) and by LEAF (images landing in the leaf); Wilson CIs;
min group size 5; overall agreement reported alongside (the number
that would hide per-reason variation).

Also: per-class top-|coefficient| features of the linear head vs. the
literals of the reason patterns that force that class (does the linear
model use the same concepts the certificate names?).

PRE-REGISTERED PREDICTIONS (before the run):
  CF1 overall tree-vs-logreg agreement at K=4 in 0.70-0.90.
  CF2 the certified mechanism pattern {bill_color=yellow = 0} (the
     pipit single clause, n=30) has conditional fidelity >= 0.90 --
     the certificate transfers to the linear model.
  CF3 multi-clause patterns show lower and more variable conditional
     fidelity than single-clause patterns.
  CF4 the pipit class's dominant linear coefficient is the
     bill_color=yellow feature (same concept the certificate names).
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="4,10")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--min-group", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    dtr = np.load(Path(args.cache_dir) / "train_features.npz")
    dte = np.load(Path(args.cache_dir) / "test_features.npz")
    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=args.seed)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=args.seed)
    vocab = ds_train.bound_attributes
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=list(vocab))
    feat_names = compiler.feature_names
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    part_bits = [n for n in feat_names if n.startswith("has_part(")]

    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])

    def pooled(rows, fmaps):
        out = {}
        fm = torch.from_numpy(fmaps.astype(np.float32))
        for i, row in enumerate(rows):
            sid, _m, boxes, vis, _a = row
            regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                    for p in ds_train.part_names if vis.get(p, 0.0) > 0.5 and p in boxes]
            if regs:
                p = pool_region_features(fm[i:i + 1], regs)
                out[int(sid)] = {r.part_name: p[r.id].numpy() for r in regs}
        return out

    print(f"--> pooling train {len(tr_rows)} / test {len(te_rows)}", flush=True)
    tr_pooled = pooled(tr_rows, dtr["feature_maps"])
    te_pooled = pooled(te_rows, dte["feature_maps"])
    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

    sids = np.array(sorted(tr_pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(args.val_fraction * len(sids))
    fit_sids = set(sids[perm[n_val:]].tolist())
    reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
    region_rows = defaultdict(list)
    for sid in fit_sids:
        for part, vec in tr_pooled[sid].items():
            region_rows[part].append((sid, vec))
    region_data = {}
    for part, rows in region_rows.items():
        cols = attr_by_part.get(part)
        if not cols:
            continue
        X = torch.from_numpy(np.stack([v for _, v in rows])).float()
        y = torch.from_numpy(np.array(
            [tr_gt[s][cols] for s, _ in rows], dtype=np.float32))
        pos, neg = y.sum(dim=0), y.shape[0] - y.sum(dim=0)
        pw = (neg / pos.clamp(min=1.0)).clamp(max=50.0)
        region_data[part] = (X, y, pw, cols)
    for _ in range(args.epochs):
        loss = 0.0
        for X, y, pw, cols in region_data.values():
            W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
            b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
            loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    print("--> readers trained", flush=True)

    def vec_from(parts):
        d = {}
        for part, vec in parts.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                with torch.no_grad():
                    t = torch.from_numpy(vec.reshape(1, -1)).float()
                    p = float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
                d[f"{a.part}:{a.attr_name}={a.value}"] = 1 if p >= 0.5 else 0
        for n in part_bits:
            d[n] = 1
        return np.array([d.get(n, 0) for n in feat_names], dtype=np.int64)

    tr_vec = {s: vec_from(tr_pooled[s]) for s in sorted(fit_sids)}
    te_vec = {s: vec_from(te_pooled[s]) for s in te_pooled.keys()}

    results = []
    for K in classes:
        tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
        te_ids = [s for s in te_vec if sid_to_label_te[s] < K]
        X_fit = np.stack([tr_vec[s] for s in tr_ids])
        y_fit = np.array([sid_to_label_tr[s] for s in tr_ids])
        X_te = np.stack([te_vec[s] for s in te_ids])
        y_te = np.array([sid_to_label_te[s] for s in te_ids])

        dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                    random_state=42).fit(X_fit, y_fit)
        head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                                   feature_names=feat_names)
        head.clf, head.is_fitted = dt, True
        backend = PyXAIBackend(head)
        lr_model = LogisticRegression(max_iter=2000, C=1.0,
                                      random_state=42).fit(X_fit, y_fit)
        tree_pred = dt.predict(X_te)
        neural_pred = lr_model.predict(X_te)
        overall_agree = float(np.mean(tree_pred == neural_pred))
        tree_acc = float(np.mean(tree_pred == y_te))
        neural_acc = float(np.mean(neural_pred == y_te))
        print(f"--> K={K}: tree acc {tree_acc:.3f} | logreg acc {neural_acc:.3f} | "
              f"overall agreement {overall_agree:.3f}", flush=True)

        # reason patterns
        patterns = {}
        for i, s in enumerate(te_ids):
            pred = int(tree_pred[i])
            exp = backend.explain_sufficient(X_te[i], pred, sample_id=s)
            lits = tuple(sorted((int(c.feature_idx), 1 if c.operator != "<=" else 0)
                                for c in exp.conditions))
            patterns.setdefault(lits, pred)
        pattern_rows = []
        for lits, c in patterns.items():
            mask = np.all(X_te[:, [f for f, _ in lits]] ==
                          np.array([v for _, v in lits]), axis=1)
            n = int(mask.sum())
            if n < args.min_group:
                continue
            k = int((neural_pred[mask] == c).sum())
            pattern_rows.append({
                "literals": [f"{feat_names[f]}{'+' if v else '-'}" for f, v in lits],
                "tree_class": int(c), "n_satisfying": n,
                "conditional_fidelity": k / n, "wilson95": wilson(k, n),
            })
        pattern_rows.sort(key=lambda r: -r["n_satisfying"])
        # per-leaf
        leaf_ids = dt.apply(X_te)
        leaf_rows = []
        for lf in sorted(set(leaf_ids)):
            mask = leaf_ids == lf
            n = int(mask.sum())
            if n < args.min_group:
                continue
            c = int(dt.predict(X_te[mask][0:1])[0])
            k = int((neural_pred[mask] == c).sum())
            leaf_rows.append({"leaf": int(lf), "tree_class": c, "n": n,
                              "conditional_fidelity": k / n, "wilson95": wilson(k, n)})
        # linear coefficients: per class, top-5 |coef|
        coef_top = {}
        for ci, cls in enumerate(lr_model.classes_):
            top = np.argsort(-np.abs(lr_model.coef_[ci]))[:5]
            coef_top[int(cls)] = [f"{feat_names[j]}:{lr_model.coef_[ci][j]:+.2f}"
                                  for j in top]
        results.append({
            "K": K, "n_test": len(te_ids),
            "tree_acc": tree_acc, "logreg_acc": neural_acc,
            "overall_agreement": overall_agree,
            "pattern_rows": pattern_rows, "leaf_rows": leaf_rows,
            "logreg_top_coefs": coef_top,
        })

    report = {
        "protocol": {
            "question": ("Conditional fidelity: P(linear-head(x) = c | phi(x) "
                         "satisfies R) for certified reasons R of the committed "
                         "trees; reference = multinomial logreg on the same "
                         "predicted concept bits"),
            "predictions_pre_registered": [
                "CF1 overall agreement at K=4 in 0.70-0.90",
                "CF2 {bill_color=yellow=0} pattern fidelity >= 0.90",
                "CF3 multi-clause patterns lower/more variable than single-clause",
                "CF4 pipit's dominant logreg coefficient is bill_color=yellow",
            ],
            "min_group": args.min_group,
            "note": ("pattern grouping satisfies-literals mask over ALL test "
                     "images of the K classes; Wilson per group; multiplicity "
                     "not corrected here (exploratory)"),
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
