#!/usr/bin/env python3
"""Task 3 (due day 4): the forcing sweep -- why does trusted-only report 0.07-0.10?

Self-contained re-run of the audit protocol (linear readers on pooled DINOv3
region features, val-carve-out isotonic per concept, CART T depth12 on reader
bits -- seed 42), then per test instance:
  - irreducible (greedy) reason from the decision path via FastBinaryTree.forces
  - delta sweep on polarity-aware calibrated p_true (val-fit ONLY):
      kept      = reason literals with p_true >= delta
      forcing   = kept subset forces the prediction (the guarantee_scaling event)
      minimal   = greedy irreducible subset of kept
      ranking   = ALL reason literals >= delta (the abstention_curves rule)
  - support scored on kept, minimal, and the whole reason, vs CUB annotations.

Deliverable: threshold table per K (4, 10) with coverage + both support numbers
+ ranking comparison, every percentage with its n, per-instance dump, plus a
side-by-side with the committed guarantee_scaling trusted-only numbers.
Minimality here is greedy-irreducible (documented), not exact-enumeration.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from grounded_vision.data.cub import CUBDataset  # noqa: E402
from grounded_vision.perception.regions import Region, pool_region_features  # noqa: E402
from grounded_vision.perception.readers import AttributeReader  # noqa: E402
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode  # noqa: E402
from run_cbm_cem_audit import FastBinaryTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DELTAS = [0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=common.DEFAULT_CACHE)
    ap.add_argument("--data-dir", default=common.DEFAULT_DATA)
    ap.add_argument("--ks", default="4,10")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    checks = []

    dtr = np.load(os.path.join(args.cache_dir, "train_features.npz"))
    dte = np.load(os.path.join(args.cache_dir, "test_features.npz"))
    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=42)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=42)
    vocab = ds_train.bound_attributes
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=list(vocab))
    feat_names = compiler.feature_names
    fidx = compiler.feature_to_idx
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    part_feat_idx = [fidx[n] for n in feat_names if n.startswith("has_part(")]
    feat_part = {fidx[f"{a.part}:{a.attr_name}={a.value}"]: a.part for a in vocab}
    for n in feat_names:
        if n.startswith("has_part("):
            feat_part[fidx[n]] = n[len("has_part("):-1]

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

    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    te_gt = {int(m[0]): m[4] for m in te_rows}
    te_vis = {int(m[0]): m[3] for m in te_rows}
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

    rng = np.random.RandomState(42)
    torch.manual_seed(42)
    print("pooling ...", flush=True)
    tr_pooled = pooled(tr_rows, dtr["feature_maps"])
    te_pooled = pooled(te_rows, dte["feature_maps"])
    sids = np.array(sorted(tr_pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(0.15 * len(sids))
    val_sids = set(sids[perm[:n_val]].tolist())
    fit_sids = set(sids[perm[n_val:]].tolist())

    # ---- readers (linear; identical protocol to the audit) -------------------
    reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
    opt = torch.optim.Adam(reader.parameters(), lr=1e-3)
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
        y = torch.from_numpy(np.array([tr_gt[s][cols] for s, _ in rows], dtype=np.float32))
        pos, neg = y.sum(dim=0), y.shape[0] - y.sum(dim=0)
        pw = (neg / pos.clamp(min=1.0)).clamp(max=50.0)
        region_data[part] = (X, y, pw, cols)
    for _ in range(args.epochs):
        loss = 0.0
        for part, (X, y, pw, cols) in region_data.items():
            W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
            b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
            loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    print("readers trained", flush=True)

    def scores_map(parts):
        d = {}
        for part, vec in parts.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                t = torch.from_numpy(vec.reshape(1, -1)).float()
                with torch.no_grad():
                    lg = reader.classifiers[f"attr_{j}"](t)
                d[f"{a.part}:{a.attr_name}={a.value}"] = float(torch.sigmoid(lg)[0, 0])
        return d

    # ---- val-only isotonic calibrators (polarity-aware p_true) ---------------
    val_scores = defaultdict(list)
    for sid in val_sids:
        for name, sc in scores_map(tr_pooled[sid]).items():
            j = next(k for k, a in enumerate(vocab)
                     if f"{a.part}:{a.attr_name}={a.value}" == name)
            val_scores[fidx[name]].append((sc, float(tr_gt[sid][j])))
    isotonic = {}
    for fi, pairs in val_scores.items():
        if len(pairs) < 10:
            continue
        xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
        if ys.min() == ys.max():
            isotonic[fi] = ("const", float(ys[0]))
        else:
            isotonic[fi] = ("ir", IsotonicRegression(out_of_bounds="clip").fit(xs, ys))

    def cal_prob(fi, s):
        if fi not in isotonic:
            return s
        kind, obj = isotonic[fi]
        return float(obj) if kind == "const" else float(obj.predict([s])[0])

    def p_true(fi, bit, s):
        if fi in part_feat_idx:
            return 1.0
        cal = cal_prob(fi, s)
        return cal if bit == 1 else 1.0 - cal

    def bits_and_scores(parts, vis):
        d = scores_map(parts)
        sc = np.zeros(len(feat_names), dtype=np.float64)
        for name, p in d.items():
            sc[fidx[name]] = p
        bits = (sc >= 0.5).astype(np.int64)
        for fi in part_feat_idx:
            bits[fi] = 1 if vis.get(feat_part[fi], 0.0) > 0.5 else 0
        return bits, sc

    def gt_vec(gt, vis):
        b = {f"{a.part}:{a.attr_name}={a.value}": float(gt[j]) for j, a in enumerate(vocab)}
        v = np.zeros(len(feat_names), dtype=np.int64)
        for name, val in b.items():
            v[fidx[name]] = int(val)
        for fi in part_feat_idx:
            v[fi] = 1 if vis.get(feat_part[fi], 0.0) > 0.5 else 0
        return v

    def path_literals(dt, fb, x, pred):
        ind = dt.decision_path(x.reshape(1, -1))
        nodes = set(ind.indices[ind.indptr[0]:ind.indptr[1]])
        t = dt.tree_
        lits, nid = [], 0
        while t.children_left[nid] != -1:
            f = int(t.feature[nid])
            left = t.children_left[nid] in nodes
            lits.append((f, 0 if left else 1))
            nid = t.children_left[nid] if left else t.children_right[nid]
        assign = dict(lits)
        assert fb.forces(assign, pred), "full path must force the prediction"
        # greedy irreducible subset (documented: not exact enumeration)
        kept = list(lits)
        i = 0
        while i < len(kept):
            trial = kept[:i] + kept[i + 1:]
            if trial and fb.forces(dict(trial), pred):
                kept.pop(i)
            else:
                i += 1
        return lits, kept

    out_blocks = []
    for K in [int(k) for k in args.ks.split(",")]:
        tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
        te_ids = [s for s in te_pooled.keys() if sid_to_label_te[s] < K]
        y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
        # has_part bits for train: every pooled part was visible by construction,
        # so visibility = 1 for pooled parts is exact here.
        X_tr = np.stack([bits_and_scores(tr_pooled[s],
                                         {p: 1.0 for p in tr_pooled[s]})[0]
                         for s in tr_ids])
        dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                    random_state=42).fit(X_tr, y_tr)
        fb = FastBinaryTree(dt)
        rows = []
        for s in sorted(te_ids):
            bits, sc = bits_and_scores(te_pooled[s], te_vis[s])
            xg = gt_vec(te_gt[s], te_vis[s])
            pred = int(dt.predict(bits.reshape(1, -1))[0])
            _, reason = path_literals(dt, fb, bits, pred)
            rec = {"sid": int(s), "pred": pred, "true": int(sid_to_label_te[s]),
                   "len_reason": len(reason),
                   "literals": [{"f": int(f), "b": int(b),
                                 "p_true": round(p_true(f, b, sc[f]), 4),
                                 "supported": bool(int(xg[f]) == b)}
                                for f, b in reason],
                   "deltas": {}}
            for d in DELTAS:
                kept = [(l["f"], l["b"]) for l in rec["literals"] if l["p_true"] >= d]
                forcing = bool(kept) and fb.forces(dict(kept), pred)
                minimal = list(kept)
                i = 0
                while i < len(minimal):
                    trial = minimal[:i] + minimal[i + 1:]
                    if trial and fb.forces(dict(trial), pred):
                        minimal.pop(i)
                    else:
                        i += 1
                sup = [l["supported"] for l in rec["literals"]]
                sup_kept = [bool(int(xg[f]) == b) for f, b in kept] if kept else None
                sup_min = [bool(int(xg[f]) == b) for f, b in minimal] if minimal else None
                rec["deltas"][str(d)] = {
                    "n_kept": len(kept), "forcing": forcing,
                    "support_kept": (all(sup_kept) if sup_kept is not None else None),
                    "n_minimal": len(minimal),
                    "support_minimal": (all(sup_min) if sup_min is not None else None),
                    "ranking_cover": all(l["p_true"] >= d for l in rec["literals"]),
                    "reason_all_supported": all(sup) if sup else None,
                }
            rows.append(rec)

        n = len(rows)
        table = {}
        print(f"\n== K={K} (n={n}) ==")
        print(f"{'delta':>6} {'cov_forc':>16} {'sup_kept':>16} {'sup_min':>16} "
              f"{'cov_rank':>16} {'sup_rank':>16}")
        for d in DELTAS:
            cells = [r["deltas"][str(d)] for r in rows]
            cov_f = sum(c["forcing"] for c in cells)
            sk = [c["support_kept"] for c in cells if c["forcing"] and
                  c["support_kept"] is not None]
            sm = [c["support_minimal"] for c in cells if c["forcing"] and
                  c["support_minimal"] is not None]
            cov_r = sum(c["ranking_cover"] for c in cells)
            sr = [c["reason_all_supported"] for c in cells if c["ranking_cover"]]
            table[str(d)] = {
                "coverage_forcing": common.n_pct(cov_f, n),
                "support_kept_given_forcing": common.n_pct(sum(sk), len(sk)) if sk else None,
                "support_minimal_given_forcing": common.n_pct(sum(sm), len(sm)) if sm else None,
                "coverage_ranking": common.n_pct(cov_r, n),
                "support_reason_given_ranking": common.n_pct(sum(sr), len(sr)) if sr else None,
            }
            print(f"{d:>6} {table[str(d)]['coverage_forcing']:>16} "
                  f"{(table[str(d)]['support_kept_given_forcing'] or '-'):>16} "
                  f"{(table[str(d)]['support_minimal_given_forcing'] or '-'):>16} "
                  f"{table[str(d)]['coverage_ranking']:>16} "
                  f"{(table[str(d)]['support_reason_given_ranking'] or '-'):>16}")
        out_blocks.append({"K": K, "n": n, "table": table, "per_instance": rows})
        checks.append(common.check(f"K={K}: per-instance dump ({n} rows, {len(DELTAS)} deltas)",
                                   n > 0))

    # side-by-side with the committed scaling numbers
    gs_path = os.path.join(REPO, "results", "guarantee_scaling_cub.json")
    if os.path.exists(gs_path):
        gs = json.load(open(gs_path))
        print("\n== committed guarantee_scaling trusted-only (per_instance arm) ==")
        for e in gs["sweep"]:
            if e["n_classes"] not in [int(k) for k in args.ks.split(",")]:
                continue
            for d, cell in e.get("trusted_only", {}).items():
                pi = cell.get("per_instance", {})
                print(f"K={e['n_classes']} delta={d}: coverage "
                      f"{pi.get('coverage')} (n={pi.get('n_covered')}), "
                      f"P(kept true|covered)={pi.get('p_kept_true_given_covered')}")
        checks.append(common.check("side-by-side printed vs guarantee_scaling", True))

    out = common.out_path("task3_forcing_sweep.json")
    json.dump({"deltas": DELTAS, "prereg": "E1-E3 in CVPR2027_PREREG_TASK_RUNS.md",
               "blocks": out_blocks}, open(out, "w"), indent=1)
    checks.append(common.check(f"wrote {os.path.basename(out)} (interpretation stays with the lead)", True))
    common.finish(checks)


if __name__ == "__main__":
    main()
