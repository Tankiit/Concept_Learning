"""
Script: run_certified_only_cub.py
==================================
The certified-concepts experiment (follow-up review, 2026-11-08):

Retrain the concept tree on ONLY the concepts whose grounding passed
milestone v2 specificity under ALL THREE control families, then rerun
the §8 all-clauses-correct measurement at K=4.  If P(all clauses
GT-correct) jumps well above the committed 25.4% (k~3.6), the
certifiable-concepts story is real; if not, the gap is perception and
the Option-A story stands.

Arms (features of the tree):
- full36          sanity: reproduces guarantee_scaling_cub.json K=4
- reader_cert4    passes all 3 control families under the DEPLOYED
                  reader instrument (predicate_grounding_milestone_v2)
- probe_cert7     passes all 3 control families under the milestone
                  probes (reader instrument partially fails these)

Protocol identical to scripts/run_guarantee_scaling_cub.py (seed 42,
readers on the 85% sub-split, CART depth 12 / leaf 4, binary unknown->0
encoding) so the full36 arm is a byte-level sanity check of the fork.
has_part bits are EXCLUDED from the certified arms (protocol-trivial:
predicted bit is always 1 by construction).

Also computed per arm, for the bound question:
- per-concept PER-POLARITY clause correctness (bank-level precision
  with Wilson 95% intervals, in-sample on this test set);
- independence product bound and the correlation-robust union
  (Boole/Frechet) lower bound 1 - sum(1 - p_i), evaluated per reason;
- observed P(all correct) vs both bounds, and per-k breakdown.
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
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


# Certification sets: concepts passing specificity (>= 1.5x target-vs-control,
# CI lower > 1) under ALL of adjacent / farthest / dist-matched controls,
# per instrument, from results/predicate_grounding_milestone_v2.json.
READER_CERTIFIED = [
    "beak:bill_shape=dagger",
    "beak:bill_color=yellow",
    "belly:belly_color=white",
    "belly:belly_color=black",
]
PROBE_CERTIFIED = READER_CERTIFIED + [
    "beak:bill_shape=cone",
    "beak:bill_color=black",
    "belly:belly_color=yellow",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    K = args.k_classes
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

    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    te_gt = {int(m[0]): m[4] for m in te_rows}
    te_vis = {int(m[0]): m[3] for m in te_rows}
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

    # ---- readers on the 85% sub-split (identical to the scaling script) ------
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
        d = {}
        for part, vec in pooled.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                p = prob_attr(j, vec)
                d[f"{a.part}:{a.attr_name}={a.value}"] = (1 if p >= 0.5 else 0, p)
        for n in part_bits:
            d[n] = (1, 1.0)  # GT-region protocol: visibility is annotation
        return d

    def vec_from(d, names):
        return np.array([d.get(n, (0, 0.0))[0] for n in names], dtype=np.int64)

    def gt_vec(gt, vis, names):
        bits = {f"{a.part}:{a.attr_name}={a.value}": float(gt[j])
                for j, a in enumerate(vocab)}
        for n in part_bits:
            p = n[len("has_part("):-1]
            bits[n] = 1.0 if vis.get(p, 0.0) > 0.5 else 0.0
        return np.array([bits.get(n, 0.0) for n in names], dtype=np.int64)

    tr_cp = {sid: concept_probs(tr_pooled[sid]) for sid in sids}
    te_cp = {sid: concept_probs(te_pooled[sid]) for sid in te_pooled.keys()}

    tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
    te_ids = [s for s in te_pooled.keys() if sid_to_label_te[s] < K]
    print(f"--> K={K}: {len(tr_ids)} train / {len(te_ids)} test images", flush=True)

    arms = [
        ("full36", list(feat_names)),
        ("reader_cert4", READER_CERTIFIED),
        ("probe_cert7", PROBE_CERTIFIED),
    ]

    results = []
    for arm_name, names in arms:
        cols_idx = [fidx[n] for n in names]
        X_fit = np.stack([vec_from(tr_cp[s], names) for s in tr_ids])
        y_fit = np.array([sid_to_label_tr[s] for s in tr_ids])
        dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                    random_state=42).fit(X_fit, y_fit)
        head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                                   feature_names=names)
        head.clf, head.is_fitted = dt, True
        backend = PyXAIBackend(head)

        k_list, acc_hits, all_correct = [], 0, []
        wrong_counts, binom_vars = [], []
        present_c, present_t, absent_c, absent_t = 0, 0, 0, 0
        per_stat = {(n, pol): [0, 0] for n in names for pol in ("pos", "neg")}
        per_k = {}
        reason_rows = []
        for sid in te_ids:
            xp = vec_from(te_cp[sid], names)
            xg = gt_vec(te_gt[sid], te_vis[sid], names)
            pred = int(dt.predict(xp.reshape(1, -1))[0])
            acc_hits += int(pred == sid_to_label_te[sid])
            try:
                exp = backend.explain_sufficient(xp, pred, sample_id=sid)
            except Exception as e:  # degenerate reasons on tiny trees
                print(f"    (explain failed sid={sid}: {e})", flush=True)
                continue
            lits = [(-1 if c.operator == "<=" else 1) * (c.feature_idx + 1)
                    for c in exp.conditions]
            k_list.append(len(lits))
            wrong = 0
            rates = []
            for lit in lits:
                f = abs(lit) - 1
                want = 1 if lit > 0 else 0
                good = bool(xg[f] == want)
                wrong += not good
                key = (names[f], "pos" if lit > 0 else "neg")
                per_stat[key][0] += good
                per_stat[key][1] += 1
                if lit > 0:
                    present_t += 1; present_c += good
                else:
                    absent_t += 1; absent_c += good
                c0, n0 = per_stat[key]
                rates.append(c0 / max(1, n0))
            all_correct.append(wrong == 0)
            wrong_counts.append(wrong)
            binom_vars.append(sum(p * (1 - p) for p in rates))
            per_k.setdefault(len(lits), [0, 0])
            per_k[len(lits)][0] += (wrong == 0)
            per_k[len(lits)][1] += 1
            reason_rows.append({"sid": sid, "k": len(lits),
                                "all_correct": bool(wrong == 0)})

        p_all = float(np.mean(all_correct)) if all_correct else None
        obs_var = float(np.var(wrong_counts)) if len(wrong_counts) > 1 else None
        bin_var = float(np.mean(binom_vars)) if binom_vars else None

        # ---- bounds: product (independence) vs union (correlation-robust) ----
        # per-concept per-polarity Wilson-95%-lower rates, then per-reason bounds.
        rate_lo, rate_hat = {}, {}
        for (n, pol), (c0, n0) in per_stat.items():
            if n0:
                rate_hat[(n, pol)] = c0 / n0
                w = wilson(c0 / n0, n0)
                rate_lo[(n, pol)] = w[0]
        prod_list, union_list, union_lo_list = [], [], []
        for sid in te_ids:
            xp = vec_from(te_cp[sid], names)
            pred = int(dt.predict(xp.reshape(1, -1))[0])
            try:
                exp = backend.explain_sufficient(xp, pred, sample_id=sid)
            except Exception:
                continue
            lits = [(-1 if c.operator == "<=" else 1) * (c.feature_idx + 1)
                    for c in exp.conditions]
            ps_hat = [rate_hat.get((names[abs(l) - 1],
                                    "pos" if l > 0 else "neg"), 1.0) for l in lits]
            ps_lo = [rate_lo.get((names[abs(l) - 1],
                                  "pos" if l > 0 else "neg"), 1.0) for l in lits]
            prod_list.append(float(np.prod(ps_hat)))
            union_list.append(max(0.0, 1.0 - sum(1.0 - p for p in ps_hat)))
            union_lo_list.append(max(0.0, 1.0 - sum(1.0 - p for p in ps_lo)))

        entry = {
            "arm": arm_name,
            "features": names,
            "n_features": len(names),
            "n_train": len(tr_ids), "n_test": len(te_ids),
            "n_reasons": len(k_list),
            "n_zero_condition_reasons": int(sum(1 for k in k_list if k == 0)),
            "n_tree_nodes": int(dt.tree_.node_count),
            "test_accuracy": acc_hits / len(te_ids),
            "mean_reason_size": float(np.mean(k_list)),
            "p_all_clauses_correct": p_all,
            "p_all_wilson95": wilson(p_all, len(all_correct)) if p_all is not None else None,
            "present_clause_precision": present_c / max(1, present_t),
            "present_clauses": present_t,
            "absent_clause_precision": absent_c / max(1, absent_t),
            "absent_clauses": absent_t,
            "correlation_test": {
                "observed_var_wrong_clauses": obs_var,
                "binomial_predicted_var": bin_var,
                "ratio": obs_var / bin_var if bin_var else None,
            },
            "per_k": {str(k): {"p_all": v[0] / v[1], "n": v[1]}
                      for k, v in sorted(per_k.items())},
            "per_concept_polarity": {
                f"{n}|{pol}": {"correct": c0, "n": n0,
                               "rate": c0 / n0 if n0 else None,
                               "wilson95": wilson(c0 / n0, n0) if n0 else None}
                for (n, pol), (c0, n0) in sorted(per_stat.items()) if n0},
            "bounds": {
                "note": ("in-sample on this test set; product bound assumes "
                         "independence and is NOT valid under positive error "
                         "correlation; union/Boole bound holds under any "
                         "dependence"),
                "mean_product_bound": float(np.mean(prod_list)) if prod_list else None,
                "mean_union_bound": float(np.mean(union_list)) if union_list else None,
                "mean_union_bound_wilson_lo": float(np.mean(union_lo_list)) if union_lo_list else None,
                "observed_p_all": p_all,
                "union_bound_holds": (p_all >= float(np.mean(union_lo_list))
                                      if union_lo_list and p_all is not None else None),
            },
        }
        results.append(entry)
        print(f"--> arm {arm_name:12s} | acc {entry['test_accuracy']:.3f} | "
              f"k {entry['mean_reason_size']:.2f} | P(all) "
              f"{p_all if p_all is not None else float('nan'):.3f} "
              f"{entry['p_all_wilson95']} | present-prec "
              f"{entry['present_clause_precision']:.3f} | absent-prec "
              f"{entry['absent_clause_precision']:.3f} | var-ratio "
              f"{entry['correlation_test']['ratio']}", flush=True)

    report = {
        "protocol": {
            "question": ("Does restricting the tree to concepts certified by "
                         "milestone-v2 specificity (all three control families) "
                         "raise P(all clauses GT-correct) at K=4?"),
            "baseline": ("guarantee_scaling_cub.json K=4: acc 0.6017, k 3.61, "
                         "P(all) 0.2542 [0.184, 0.340]"),
            "certification_source": "results/predicate_grounding_milestone_v2.json",
            "certification_rule": ("passes_by_control true for adjacent AND "
                                   "farthest AND dist_matched, per instrument"),
            "reader_certified": READER_CERTIFIED,
            "probe_certified": PROBE_CERTIFIED,
            "cache": "eval-transform (flip-free)",
            "tree": "CART depth 12, leaf 4, seed 42",
            "encoding": "binary unknown->0 (pre-tristate, matches the committed sweep)",
            "has_part": "excluded from certified arms (predicted bit trivially 1)",
            "in_sample_caveat": ("per-concept rates and bounds are computed on "
                                 "the same test reasons they bound (deciding "
                                 "experiment, not out-of-sample certificate)"),
        },
        "arms": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
