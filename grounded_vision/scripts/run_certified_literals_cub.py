"""
Script: run_certified_literals_cub.py
======================================
Certified LITERALS, selected out-of-sample (follow-up review round 2,
2026-11-08). The unit of certification is (concept, polarity), not the
concept: a literal is certified iff, on the VALIDATION sub-split only,
its assertion is GT-correct with Wilson-95%-lower >= tau over >=
min_support assertions.  Enforcement is at inference (the CART-native
per-polarity veto needs a constrained learner -- stated as future work):

  mechanism A (certified minimal reason): keep the native minimal
    sufficient reason only if every literal in it is certified;
    otherwise abstain.
  mechanism B (certified forcing set): keep the instance's certified
    literal values iff they ALONE force the tree's prediction (exact
    reachable-leaf DFS); report a greedily-minimal such set.

Trees: the committed full-36 CART (depth 12, leaf 4, seed 42) per K in
{4, 10} -- identical substrate to guarantee_scaling_cub.json.

PRE-REGISTERED PREDICTIONS (written before the run; do not edit):
  P1 (K=4 structure): the 30/118 all-correct minimal reasons concentrate
     on 1-2 species and essentially one single-clause leaf condition.
  P2 (certification): (beak:bill_color=yellow, pos) fails at every tau;
     most present-literals fail at tau=0.8; most absent-literals pass.
  P3 (K=4, mech A): coverage 10-40%; P(all clauses correct | covered)
     70-100%; union bound (val Wilson-lower) <= observed.
  P4 (K=4, mech B): coverage 30-70%; mean kept 4-10 literals;
     P(kept all true | covered) 40-80%.
  P5 (K=10, mech A): coverage < 15% (longer reasons -> more literals);
     P(all | covered) still >= 70%.
  P6: union bound <= observed at every K and mechanism (it can only
     fail if validation rates overestimate test rates).
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
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


class FastBinaryTree:
    def __init__(self, dt):
        t = dt.tree_
        self.f = t.feature.astype(np.int64)
        self.l = t.children_left.astype(np.int64)
        self.r = t.children_right.astype(np.int64)
        self.pred = np.array(
            [int(dt.classes_[int(np.argmax(t.value[i][0]))])
             for i in range(t.node_count)], dtype=np.int64)

    def forces(self, assign, target):
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
    ap.add_argument("--classes", default="4,10")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--taus", default="0.7,0.8,0.9")
    ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
    taus = [float(t) for t in args.taus.split(",")]
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
    name_to_j = {f"{a.part}:{a.attr_name}={a.value}": j for j, a in enumerate(vocab)}
    part_bits = [n for n in feat_names if n.startswith("has_part(")]

    # species names for the per-species dump
    img_root = Path(args.data_dir) / "images"
    class_names = sorted([d.name.split(".", 1)[1] for d in img_root.iterdir()
                          if not d.name.startswith(".")])

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
    print(f"--> readers trained on {len(fit_sids)} imgs (validation {len(val_sids)})",
          flush=True)

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
            d[n] = (1, 1.0)
        return d

    def vec_from(d):
        return np.array([d.get(n, (0, 0.0))[0] for n in feat_names], dtype=np.int64)

    def gt_bits(gt, vis):
        bits = {f"{a.part}:{a.attr_name}={a.value}": int(gt[j])
                for j, a in enumerate(vocab)}
        for n in part_bits:
            p = n[len("has_part("):-1]
            bits[n] = 1 if vis.get(p, 0.0) > 0.5 else 0
        return bits

    tr_cp = {sid: concept_probs(tr_pooled[sid]) for sid in sids}
    te_cp = {sid: concept_probs(te_pooled[sid]) for sid in te_pooled.keys()}

    # ---- literal certification on the VALIDATION sub-split only ---------------
    lit_stat = {}  # (name, pol) -> [correct, n]
    for sid in val_sids:
        # part bits are asserted only where the part was read (pooled), so the
        # GT-visibility reference marks exactly those parts visible
        gtb = gt_bits(tr_gt[sid], {p: 1.0 for p in tr_pooled[sid]})
        for name, (bit, _p) in tr_cp[sid].items():
            key = (name, bit)
            lit_stat.setdefault(key, [0, 0])
            lit_stat[key][1] += 1
            lit_stat[key][0] += int(gtb.get(name, None) == bit)
    cert_table = {}
    for (name, pol), (c0, n0) in sorted(lit_stat.items()):
        w = wilson(c0, n0)
        cert_table[f"{name}|{'pos' if pol else 'neg'}"] = {
            "n_val": n0, "correct": c0, "rate": c0 / n0 if n0 else None,
            "wilson_lo": w[0] if w else None,
            "certified_at": {str(t): bool(n0 >= args.min_support and w and w[0] >= t)
                             for t in taus},
        }
    for t in taus:
        ncert = sum(1 for v in cert_table.values() if v["certified_at"][str(t)])
        print(f"--> tau={t}: {ncert}/{len(cert_table)} literals certified "
              f"(support>={args.min_support})", flush=True)

    # ---- per-K evaluation ------------------------------------------------------
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

        per_image = []
        mechA = {t: {"covered": 0, "all_correct": 0, "k_sizes": [], "bounds": []}
                 for t in taus}
        mechB = {t: {"covered": 0, "all_correct": 0, "kept_sizes": [], "bounds": []}
                 for t in taus}
        for sid in te_ids:
            cp = te_cp[sid]
            xp = vec_from(cp)
            gtb = gt_bits(te_gt[sid], te_vis[sid])
            pred = int(dt.predict(xp.reshape(1, -1))[0])
            cls = sid_to_label_te[sid]
            exp = backend.explain_sufficient(xp, pred, sample_id=sid)
            lits = [(feat_names[c.feature_idx], 1 if c.operator != "<=" else 0)
                    for c in exp.conditions]
            row = {"sid": sid, "class": cls,
                   "k": len(lits),
                   "lits": [{"lit": f"{n}|{'pos' if p else 'neg'}",
                             "gt_correct": bool(gtb.get(n) == p)}
                            for n, p in lits],
                   "all_correct": all(gtb.get(n) == p for n, p in lits)}
            per_image.append(row)
            # mechanism A: certified minimal reason
            for t in taus:
                ok = all(cert_table.get(f"{n}|{'pos' if p else 'neg'}", {}
                                        ).get("certified_at", {}).get(str(t), False)
                         for n, p in lits)
                if ok:
                    m = mechA[t]
                    m["covered"] += 1
                    m["all_correct"] += row["all_correct"]
                    m["k_sizes"].append(len(lits))
                    m["bounds"].append(max(0.0, 1.0 - sum(
                        1.0 - cert_table[f"{n}|{'pos' if p else 'neg'}"]["wilson_lo"]
                        for n, p in lits)))
            # mechanism B: certified forcing set (greedy-minimal)
            for t in taus:
                kept = {fidx[n]: b for n, (b, _p) in cp.items()
                        if cert_table.get(f"{n}|{'pos' if b else 'neg'}", {}
                                          ).get("certified_at", {}).get(str(t), False)}
                if not kept or not fast.forces(kept, pred):
                    continue
                for f in list(kept.keys()):  # greedy drop
                    smaller = {k: v for k, v in kept.items() if k != f}
                    if smaller and fast.forces(smaller, pred):
                        kept = smaller
                m = mechB[t]
                m["covered"] += 1
                m["kept_sizes"].append(len(kept))
                good = all(gtb.get(feat_names[f]) == b for f, b in kept.items())
                m["all_correct"] += good
                m["bounds"].append(max(0.0, 1.0 - sum(
                    1.0 - cert_table[
                        f"{feat_names[f]}|{'pos' if b else 'neg'}"]["wilson_lo"]
                    for f, b in kept.items())))

        entry = {"K": K, "n_test": len(te_ids),
                 "n_train": len(tr_ids),
                 "tree_test_acc": float(np.mean(
                     [int(dt.predict(vec_from(te_cp[s]).reshape(1, -1))[0])
                      == sid_to_label_te[s] for s in te_ids])),
                 "default": {
                     "mean_k": float(np.mean([r["k"] for r in per_image])),
                     "p_all": float(np.mean([r["all_correct"] for r in per_image]))},
                 "mechanisms": {
                     f"tau={t}": {
                         "A_certified_minimal_reason": {
                             "coverage": mechA[t]["covered"] / len(te_ids),
                             "n_covered": mechA[t]["covered"],
                             "p_all_given_covered":
                                 mechA[t]["all_correct"] / mechA[t]["covered"]
                                 if mechA[t]["covered"] else None,
                             "mean_k": float(np.mean(mechA[t]["k_sizes"]))
                                 if mechA[t]["k_sizes"] else None,
                             "mean_union_bound": float(np.mean(mechA[t]["bounds"]))
                                 if mechA[t]["bounds"] else None,
                         },
                         "B_certified_forcing_set": {
                             "coverage": mechB[t]["covered"] / len(te_ids),
                             "n_covered": mechB[t]["covered"],
                             "p_all_given_covered":
                                 mechB[t]["all_correct"] / mechB[t]["covered"]
                                 if mechB[t]["covered"] else None,
                             "mean_kept": float(np.mean(mechB[t]["kept_sizes"]))
                                 if mechB[t]["kept_sizes"] else None,
                             "mean_union_bound": float(np.mean(mechB[t]["bounds"]))
                                 if mechB[t]["bounds"] else None,
                         },
                     } for t in taus},
                 }
        if K == 4:  # per-species decomposition (P1) + minimal-reason dump
            by_cls = {}
            for r in per_image:
                c = by_cls.setdefault(r["class"], {"n": 0, "all_correct": 0,
                                                   "k_hist": {}})
                c["n"] += 1
                c["all_correct"] += r["all_correct"]
                c["k_hist"][str(r["k"])] = c["k_hist"].get(str(r["k"]), 0) + 1
            entry["per_species"] = {
                class_names[c]: {"n": v["n"],
                                 "all_correct": v["all_correct"],
                                 "k_hist": v["k_hist"]}
                for c, v in sorted(by_cls.items())}
            # dominant clause among all-correct k=1 reasons
            one = [r for r in per_image if r["k"] == 1 and r["all_correct"]]
            entry["k1_correct_clauses"] = {
                c: sum(1 for r in one if r["lits"][0]["lit"] == c)
                for c in {r["lits"][0]["lit"] for r in one}}
        results.append(entry)
        dj = json.dumps(entry["mechanisms"], indent=None)
        print(f"--> K={K}: default k {entry['default']['mean_k']:.2f} "
              f"P(all) {entry['default']['p_all']:.3f} | " +
              " | ".join(
                  f"tau={t}: A cov {entry['mechanisms'][f'tau={t}']['A_certified_minimal_reason']['coverage']:.3f}"
                  f" P|cov {entry['mechanisms'][f'tau={t}']['A_certified_minimal_reason']['p_all_given_covered']}"
                  f" | B cov {entry['mechanisms'][f'tau={t}']['B_certified_forcing_set']['coverage']:.3f}"
                  f" P|cov {entry['mechanisms'][f'tau={t}']['B_certified_forcing_set']['p_all_given_covered']}"
                  for t in taus), flush=True)

    report = {
        "protocol": {
            "question": ("Certified LITERALS (concept, polarity), selected "
                         "out-of-sample on the validation sub-split, enforced "
                         "at inference; measured on test at K=4 and K=10"),
            "predictions_pre_registered": [
                "P1 K=4: 30/118 all-correct minimal reasons concentrate on 1-2 "
                "species and one single-clause leaf condition",
                "P2 (beak:bill_color=yellow, pos) fails certification at every "
                "tau; most present-literals fail at 0.8; most absent-literals pass",
                "P3 K=4 mech A: coverage 10-40%, P(all|cov) 70-100%, "
                "union bound <= observed",
                "P4 K=4 mech B: coverage 30-70%, kept 4-10, P(kept all|cov) 40-80%",
                "P5 K=10 mech A: coverage < 15%, P(all|cov) >= 70%",
                "P6 union bound <= observed everywhere",
            ],
            "certification_rule": (f"Wilson-95%-lower >= tau over >= "
                                   f"{args.min_support} validation assertions "
                                   f"of that literal (part visible)"),
            "certification_split": "15% validation sub-split of train (never test)",
            "enforcement": ("inference-time abstention; a training-time "
                            "per-polarity veto needs a constrained tree "
                            "learner (GOSDT-style literal whitelist) - future work"),
            "tree": "CART depth 12, leaf 4, seed 42, full 36 concepts",
            "cache": "eval-transform (flip-free)",
            "encoding": "binary unknown->0 (pre-tristate, matches committed sweep)",
        },
        "literal_certification": cert_table,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
