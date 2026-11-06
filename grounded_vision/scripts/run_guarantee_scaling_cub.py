"""
Script: run_guarantee_scaling_cub.py
====================================
The class-count sweep the review points to: reason LENGTH is what limits
the guarantee (P(all clauses correct) ~ p^k), so sweep the number of
classes with capped-depth trees and measure where the guarantee becomes
useful.

Per class count K in --classes (default 4,10,20,50,100,200):
- concept tree: CART depth 12 / leaf 4 on predicted 36-concept BOUND
  vectors of train images of the first K species;
- default reasons on a seeded test subsample: mean size k, accuracy,
  P(all clauses GT-correct) with Wilson CI, present/absent clause
  precision, and a DIRECT correlation test (variance of the per-reason
  wrong-clause count vs the binomial prediction);
- trusted-only coverage on ALL test images of those classes, NATIVE
  (exact reachable-leaf DFS; no PyXAI, no crashes):
  * global rule: keep the instance's literals over concepts whose
    validation accuracy >= delta;
  * per-instance rule: keep literals whose CALIBRATED probability of
    being true of THIS image >= delta (per-concept isotonic calibrators
    fit on the validation sub-split; unobserved parts -> p=0).
  Coverage = the kept literals alone force the prediction. For covered
  instances we report P(kept literals GT-true | covered) and mean kept.

Outputs: results JSON + a P(correct)-vs-k plot with independence curves.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend


def wilson(p, n, z=1.96):
    if n == 0:
        return None
    phat = p
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


class FastBinaryTree:
    """Exact reachable-leaf analysis for a binary-input sklearn tree."""

    def __init__(self, dt):
        t = dt.tree_
        self.f = t.feature.astype(np.int64)
        self.l = t.children_left.astype(np.int64)
        self.r = t.children_right.astype(np.int64)
        self.pred = np.array(
            [int(dt.classes_[int(np.argmax(t.value[i][0]))])
             for i in range(t.node_count)], dtype=np.int64)

    def forces(self, assign: dict, target: int) -> bool:
        """True iff every leaf consistent with the partial assignment
        (feature -> 0/1) predicts target, and at least one is reachable."""
        stack = [0]
        reached = False
        while stack:
            nid = stack.pop()
            if self.l[nid] == -1:
                if self.pred[nid] != target:
                    return False
                reached = True
            else:
                f = int(self.f[nid])
                if f in assign:
                    stack.append(int(self.r[nid] if assign[f] else self.l[nid]))
                else:
                    stack.append(int(self.l[nid]))
                    stack.append(int(self.r[nid]))
        return reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--classes", default="4,10,20,50,100,200")
    ap.add_argument("--n-reason-samples", type=int, default=600)
    ap.add_argument("--deltas", default="0.7,0.8,0.9")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
    deltas = [float(d) for d in args.deltas.split(",")]
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
    fidx = compiler.feature_to_idx
    attr_by_part = {}
    for j, a in enumerate(vocab):
        attr_by_part.setdefault(a.part, []).append(j)
    part_bits = [n for n in feat_names if n.startswith("has_part(")]

    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])
    print(f"--> pooling train {len(tr_rows)} / test {len(te_rows)}", flush=True)

    def pooled(rows, fmaps):
        out = {}
        fm = torch.from_numpy(fmaps.astype(np.float32))
        for i, row in enumerate(rows):
            sid, _meta_sid, boxes, vis, _attrs = row
            regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                    for p in ds_train.part_names if vis.get(p, 0.0) > 0.5 and p in boxes]
            if regs:
                pooled = pool_region_features(fm[i:i + 1], regs)
                out[int(sid)] = {r.part_name: pooled[r.id].numpy() for r in regs}
        return out

    tr_pooled = pooled(tr_rows, dtr["feature_maps"])
    te_pooled = pooled(te_rows, dte["feature_maps"])

    # rows are (sid, meta_sid, boxes, vis, attrs)
    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    te_gt = {int(m[0]): m[4] for m in te_rows}
    tr_vis = {int(m[0]): m[3] for m in tr_rows}
    te_vis = {int(m[0]): m[3] for m in te_rows}
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

    # ---- readers on the 85% sub-split ----------------------------------------
    sids = np.array(sorted(tr_pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(args.val_fraction * len(sids))
    val_sids = set(sids[perm[:n_val]].tolist())
    fit_sids = set(sids[perm[n_val:]].tolist())
    reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
    region_rows = {}
    for sid in fit_sids:
        for part, vec in tr_pooled[sid].items():
            region_rows.setdefault(part, []).append((sid, vec))
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
    for epoch in range(args.epochs):
        loss = 0.0
        for X, y, pw, cols in region_data.values():
            W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
            b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
            loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"--> readers trained on {len(fit_sids)} imgs "
          f"(validation {len(val_sids)})", flush=True)

    def prob_attr(j, vec):
        with torch.no_grad():
            t = torch.from_numpy(vec.reshape(1, -1)).float()
            return float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])

    def concept_probs(pooled):
        """{feature_name: (bit, prob)}; prob only where read."""
        d = {}
        for part, vec in pooled.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                p = prob_attr(j, vec)
                d[f"{a.part}:{a.attr_name}={a.value}"] = (1 if p >= 0.5 else 0, p)
        for n in part_bits:
            p = n[len("has_part("):-1]
            d[n] = (1, 1.0)  # GT-region protocol: visibility is annotation
        return d

    def vec_from(d):
        return np.array([d.get(n, (0, 0.0))[0] for n in feat_names], dtype=np.int64)

    def gt_vec(gt, vis):
        bits = {f"{a.part}:{a.attr_name}={a.value}": float(gt[j])
                for j, a in enumerate(vocab)}
        for n in part_bits:
            p = n[len("has_part("):-1]
            bits[n] = 1.0 if vis.get(p, 0.0) > 0.5 else 0.0
        return np.array([bits.get(n, 0.0) for n in feat_names], dtype=np.int64)

    tr_cp = {sid: concept_probs(tr_pooled[sid]) for sid in sids}
    te_cp = {sid: concept_probs(te_pooled[sid]) for sid in te_pooled.keys()}

    # ---- validation reliability + per-concept isotonic calibrators -----------
    val_acc, calibrators = {}, {}
    for j, a in enumerate(vocab):
        name = f"{a.part}:{a.attr_name}={a.value}"
        ps, ys, ok = [], [], 0
        for sid in val_sids:
            if a.part in tr_pooled[sid]:
                bit, p = tr_cp[sid][name]
                ps.append(p); ys.append(float(tr_gt[sid][j]))
                ok += int(bit == tr_gt[sid][j])
        if ps:
            val_acc[name] = ok / len(ps)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            iso.fit(np.array(ps), np.array(ys))
            calibrators[name] = iso
    for n in part_bits:
        val_acc[n] = 1.0  # GT-visibility protocol: predicted == GT bit
    trusted_global = {n for n, acc in val_acc.items() if acc >= 0.75}

    def p_true(name, bit, prob):
        """Calibrated probability that THIS literal is true of THIS image.
        Unread concepts (prob None) are never trusted."""
        if name not in calibrators:
            return 0.0 if not name.startswith("has_part") else 1.0
        cal = float(calibrators[name].predict([prob])[0])
        return cal if bit == 1 else 1.0 - cal

    # ---- class sweep ----------------------------------------------------------
    results = []
    for K in classes:
        tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
        te_ids = [s for s in te_pooled.keys() if sid_to_label_te[s] < K]
        X_fit = np.stack([vec_from(tr_cp[s]) for s in tr_ids])
        y_fit = np.array([sid_to_label_tr[s] for s in tr_ids])
        dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                    random_state=42).fit(X_fit, y_fit)
        head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                                   feature_names=feat_names)
        head.clf, head.is_fitted = dt, True
        backend = PyXAIBackend(head)
        fast = FastBinaryTree(dt)

        if len(te_ids) <= args.n_reason_samples:
            sel = list(te_ids)
        else:
            idx = np.sort(np.random.RandomState(7).permutation(len(te_ids))
                          [: args.n_reason_samples])
            sel = [te_ids[int(i)] for i in idx]

        k_list, acc_hits, all_correct = [], 0, []
        wrong_counts, binom_vars, present_c, present_t, absent_c, absent_t = \
            [], [], 0, 0, 0, 0
        per_concept_correct = {n: [0, 0] for n in feat_names}
        for sid in sel:
            xp = vec_from(te_cp[sid])
            xg = gt_vec(te_gt[sid], te_vis[sid])
            pred = int(dt.predict(xp.reshape(1, -1))[0])
            acc_hits += int(pred == sid_to_label_te[sid])
            exp = backend.explain_sufficient(xp, pred, sample_id=sid)
            lits = [(-1 if c.operator == "<=" else 1) * (c.feature_idx + 1)
                    for c in exp.conditions]
            k_list.append(len(lits))
            wrong = 0
            for lit in lits:
                f = abs(lit) - 1
                want = 1 if lit > 0 else 0
                good = bool(xg[f] == want)
                wrong += not good
                per_concept_correct[feat_names[f]][0] += good
                per_concept_correct[feat_names[f]][1] += 1
                if lit > 0:
                    present_t += 1; present_c += good
                else:
                    absent_t += 1; absent_c += good
            all_correct.append(wrong == 0)
            wrong_counts.append(wrong)
            ps = [per_concept_correct[feat_names[abs(l) - 1]][0]
                  / max(1, per_concept_correct[feat_names[abs(l) - 1]][1])
                  for l in lits]
            binom_vars.append(sum(p * (1 - p) for p in ps))
        n_ok = sum(all_correct)
        p_all = n_ok / len(all_correct)
        # direct correlation test: observed Var(W) vs mean binomial Var(W)
        obs_var = float(np.var(wrong_counts)) if len(wrong_counts) > 1 else None
        bin_var = float(np.mean(binom_vars)) if binom_vars else None

        # trusted-only (native, ALL test images of these classes)
        trust_rows = {}
        for delta in deltas:
            cov_g = cov_i = 0
            kept_g_sz = kept_i_sz = 0
            for sid in te_ids:
                cp = te_cp[sid]
                pred = int(dt.predict(vec_from(cp).reshape(1, -1))[0])
                assign_g, assign_i = {}, {}
                tg, ti = [], []
                for name, (bit, prob) in cp.items():
                    f = fidx[name]
                    if name in trusted_global:
                        assign_g[f] = bit
                        tg.append((f, bit))
                    pt = p_true(name, bit, prob)
                    if pt >= delta:
                        assign_i[f] = bit
                        ti.append((f, bit))
                if fast.forces(assign_g, pred):
                    cov_g += 1; kept_g_sz += len(tg)
                if fast.forces(assign_i, pred):
                    cov_i += 1; kept_i_sz += len(ti)
            trust_rows[delta] = {
                "global": {"coverage": cov_g / len(te_ids),
                           "mean_kept": kept_g_sz / cov_g if cov_g else None},
                "per_instance": {"coverage": cov_i / len(te_ids),
                                 "mean_kept": kept_i_sz / cov_i if cov_i else None},
            }
        # correctness of kept literals needs a second pass (GT check)
        for delta in deltas:
            row = trust_rows[delta]
            for rule, key in (("global", "global"), ("per_instance", "per_instance")):
                cov = correct = 0
                for sid in te_ids:
                    cp = te_cp[sid]
                    xg = gt_vec(te_gt[sid], te_vis[sid])
                    pred = int(dt.predict(vec_from(cp).reshape(1, -1))[0])
                    kept = []
                    for name, (bit, prob) in cp.items():
                        if rule == "global" and name in trusted_global:
                            kept.append((fidx[name], bit))
                        if rule == "per_instance" and p_true(name, bit, prob) >= delta:
                            kept.append((fidx[name], bit))
                    if fast.forces({f: b for f, b in kept}, pred):
                        cov += 1
                        correct += all(int(xg[f]) == b for f, b in kept)
                row[key]["p_kept_true_given_covered"] = correct / cov if cov else None
                row[key]["n_covered"] = cov

        entry = {
            "n_classes": K,
            "n_train": len(tr_ids), "n_test": len(te_ids),
            "n_reason_samples": len(sel),
            "test_accuracy": acc_hits / len(sel),
            "mean_reason_size": float(np.mean(k_list)),
            "p_all_clauses_correct": p_all,
            "p_all_wilson95": wilson(p_all, len(all_correct)),
            "present_clause_precision": present_c / max(1, present_t),
            "absent_clause_precision": absent_c / max(1, absent_t),
            "correlation_test": {
                "observed_var_wrong_clauses": obs_var,
                "binomial_predicted_var": bin_var,
                "ratio": obs_var / bin_var if bin_var else None},
            "trusted_only": trust_rows,
        }
        results.append(entry)
        print(f"--> K={K:3d}: acc {entry['test_accuracy']:.3f} | k "
              f"{entry['mean_reason_size']:.2f} | P(all) {p_all:.3f} "
              f"{entry['p_all_wilson95']} | var-ratio "
              f"{entry['correlation_test']['ratio']:.2f} | per-inst cov "
              + ", ".join(f"d{d}={trust_rows[d]['per_instance']['coverage']:.3f}"
                          for d in deltas), flush=True)

    # ---- plot ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = [r["mean_reason_size"] for r in results]
    ps = [r["p_all_clauses_correct"] for r in results]
    ax.plot(ks, ps, "o-", label="measured P(all clauses correct)")
    for p_clause, style in ((0.9, "--"), (0.8, ":"), (0.7, "-.")):
        ax.plot(ks, [p_clause ** k for k in ks], style, alpha=0.6,
                label=f"independence p^k, p={p_clause}")
    for r in results:
        ax.annotate(f"K={r['n_classes']}", (r["mean_reason_size"],
                                             r["p_all_clauses_correct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("mean certified reason size k (clauses)")
    ax.set_ylabel("P(all clauses GT-correct)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Guarantee scaling with reason length (CUB concept trees)")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)

    report = {
        "protocol": {
            "cache": "eval-transform", "classes": classes, "deltas": deltas,
            "tree": "CART depth 12, leaf 4, seed 42",
            "trust_global": "per-concept validation accuracy >= 0.75",
            "trust_per_instance": ("per-concept isotonic calibrator on validation; "
                                   "p_true(lit) >= delta; unobserved -> 0"),
            "native_test": "exact reachable-leaf DFS (no PyXAI)",
        },
        "sweep": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")
    print(f"saved: {args.plot}")


if __name__ == "__main__":
    main()
