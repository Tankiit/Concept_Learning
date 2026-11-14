#!/usr/bin/env python3
"""Ablation C1: tree depth -> |R| -> premise support, at fixed K.

Pre-registered as P-D1..P-D3 in CVPR2027_PREREG_TASK_RUNS.md (2026-11-14,
before this run). Protocol identical to task3_forcing_sweep (same readers,
seed 42, val carve-out, oracle boxes); only max_depth varies in
{2,4,6,8,12,16} at K in {4,10,200} (K=200: deterministic first-600 te sids).
Per cell: mean |R|, support, independence null (per-literal marginals over
that cell's reason occurrences), ratio with cluster-bootstrap 95% CI, acc.

Output: results/ablation_depth_support.json (refuses to overwrite).
Run from grounded_vision/ : PYTHONPATH=src python scripts/ablation_depth_support.py
"""
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from grounded_vision.data.cub import CUBDataset  # noqa: E402
from grounded_vision.perception.regions import Region, pool_region_features  # noqa: E402
from grounded_vision.perception.readers import AttributeReader  # noqa: E402
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode  # noqa: E402
from run_cbm_cem_audit import FastBinaryTree  # noqa: E402

CACHE = "/Users/tanmoy/research/data/grounded_vision_runs/cub_local_et"
DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
DEPTHS = [2, 4, 6, 8, 12, 16]
# Ks come from argv (default: the pre-registered 4,10,200); output file is
# named after them so reruns at other K never collide (new-files rule).
KS = [int(a) for a in (sys.argv[1:] or ["4", "10", "200"])]
OUT = os.path.join(REPO, "results", "ablation_depth_K" + "_".join(map(str, KS)) + ".json")


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING to overwrite {OUT}")
    dtr = np.load(os.path.join(CACHE, "train_features.npz"))
    dte = np.load(os.path.join(CACHE, "test_features.npz"))
    ds_train = CUBDataset(data_dir=DATA, split="train", use_mock_if_missing=False, seed=42)
    ds_test = CUBDataset(data_dir=DATA, split="test", use_mock_if_missing=False, seed=42)
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
    for _ in range(40):
        loss = 0.0
        for part, (X, y, pw, cols) in region_data.items():
            W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
            b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
            loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    print("readers trained", flush=True)

    def bits_of(parts, vis):
        sc = np.zeros(len(feat_names), dtype=np.float64)
        with torch.no_grad():
            for part, vec in parts.items():
                for j in attr_by_part.get(part, []):
                    a = vocab[j]
                    lg = reader.classifiers[f"attr_{j}"](
                        torch.from_numpy(vec.reshape(1, -1)).float())
                    sc[fidx[f"{a.part}:{a.attr_name}={a.value}"]] = float(torch.sigmoid(lg)[0, 0])
        bits = (sc >= 0.5).astype(np.int64)
        for fi in part_feat_idx:
            bits[fi] = 1 if vis.get(feat_part[fi], 0.0) > 0.5 else 0
        return bits

    def gt_vec(gt, vis):
        v = np.zeros(len(feat_names), dtype=np.int64)
        for j, a in enumerate(vocab):
            v[fidx[f"{a.part}:{a.attr_name}={a.value}"]] = int(gt[j])
        for fi in part_feat_idx:
            v[fi] = 1 if vis.get(feat_part[fi], 0.0) > 0.5 else 0
        return v

    def reason_of(dt, fb, bits, pred):
        ind = dt.decision_path(bits.reshape(1, -1))
        nodes = set(ind.indices[ind.indptr[0]:ind.indptr[1]])
        t = dt.tree_
        lits, nid = [], 0
        while t.children_left[nid] != -1:
            f = int(t.feature[nid])
            left = t.children_left[nid] in nodes
            lits.append((f, 0 if left else 1))
            nid = t.children_left[nid] if left else t.children_right[nid]
        kept = list(lits)
        i = 0
        while i < len(kept):
            trial = kept[:i] + kept[i + 1:]
            if trial and fb.forces(dict(trial), pred):
                kept.pop(i)
            else:
                i += 1
        return kept

    out = {"prereg": "P-D1..P-D3 (CVPR2027_PREREG_TASK_RUNS.md, fixed before run)",
           "cells": []}
    boot_rng = np.random.default_rng(42)
    for K in KS:
        tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
        te_ids = sorted([s for s in te_pooled.keys() if sid_to_label_te[s] < K])
        if K == 200:
            te_ids = te_ids[:600]          # deterministic, matches protocol
        y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
        X_tr = np.stack([bits_of(tr_pooled[s], {p: 1.0 for p in tr_pooled[s]}) for s in tr_ids])
        for depth in DEPTHS:
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=4,
                                        random_state=42).fit(X_tr, y_tr)
            fb = FastBinaryTree(dt)
            recs = []
            acc = 0
            for s in te_ids:
                bits = bits_of(te_pooled[s], te_vis[s])
                pred = int(dt.predict(bits.reshape(1, -1))[0])
                acc += int(pred == sid_to_label_te[s])
                xg = gt_vec(te_gt[s], te_vis[s])
                R = reason_of(dt, fb, bits, pred)
                if not R:
                    continue
                recs.append(([(f, b, bool(int(xg[f]) == b)) for f, b in R],
                             all(int(xg[f]) == b for f, b in R)))
            n = len(recs)
            supp = float(np.mean([r[1] for r in recs]))
            lens = float(np.mean([len(r[0]) for r in recs]))
            occ = defaultdict(list)
            nulls = []
            for r, _ in recs:
                for f, b, s_ in r:
                    occ[(f, b)].append(s_)
                nulls.append(float(np.prod([np.mean(occ_cur[(f, b)])
                                            for f, b, _ in r])) if False else None)
            marg = {k: float(np.mean(v)) for k, v in occ.items()}
            null = float(np.mean([np.prod([marg[(f, b)] for f, b, _ in r]) for r, _ in recs]))
            # bootstrap CI on ratio (cluster over instances, marginals re-estimated)
            ratios = []
            y_arr = np.array([r[1] for r in recs], float)
            occ_lists = [[(f, b, s_) for f, b, s_ in r] for r, _ in recs]
            for _ in range(5000):
                idx = boot_rng.integers(0, n, n)
                m = defaultdict(list)
                for i in idx:
                    for f, b, s_ in occ_lists[i]:
                        m[(f, b)].append(s_)
                mp = {k: float(np.mean(v)) for k, v in m.items()}
                nl = np.mean([np.prod([mp[(f, b)] for f, b, _ in occ_lists[i]]) for i in idx])
                ratios.append(y_arr[idx].mean() / nl if nl > 0 else np.nan)
            ratios = np.array(ratios); ratios = ratios[~np.isnan(ratios)]
            cell = {"K": K, "depth": depth, "n": n, "acc": round(acc / len(te_ids), 4),
                    "mean_len_R": round(lens, 2), "support": round(supp, 4),
                    "null": round(null, 4), "ratio": round(supp / null, 3),
                    "ratio_boot_pct": [round(float(np.percentile(ratios, 2.5)), 3),
                                       round(float(np.percentile(ratios, 97.5)), 3)]}
            out["cells"].append(cell)
            print(f"K={K:>3} depth={depth:>2}: |R|={lens:5.2f} supp={supp:.3f} "
                  f"null={null:.3f} ratio={supp/null:.2f} "
                  f"[{cell['ratio_boot_pct'][0]},{cell['ratio_boot_pct'][1]}] "
                  f"acc={cell['acc']:.3f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
