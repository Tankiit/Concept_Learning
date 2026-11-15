#!/usr/bin/env python3
"""Per-instance x per-depth dump for the K=20 depth sweep (for fig_qualitative).

Identical protocol to ablation_depth_support.py (same readers, seed 42,
K=20, depths {2,4,6,8,12,16}); adds a per-instance dump:
  sid, pred, true label, correct?, and for the reason at each depth:
  literals [(feature, bit, supported, part_visible)], all_supported, |R|.
Output: results/ablation_depth_K20_perinstance.json (refuses overwrite).
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
OUT = os.path.join(REPO, "results", "ablation_depth_K20_perinstance.json")
DEPTHS = [2, 4, 6, 8, 12, 16]
K = 20


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

    def bits_and_gt(parts, vis, gt):
        sc = np.zeros(len(feat_names), dtype=np.float64)
        with torch.no_grad():
            for part, vec in parts.items():
                for j in attr_by_part.get(part, []):
                    a = vocab[j]
                    lg = reader.classifiers[f"attr_{j}"](
                        torch.from_numpy(vec.reshape(1, -1)).float())
                    sc[fidx[f"{a.part}:{a.attr_name}={a.value}"]] = float(torch.sigmoid(lg)[0, 0])
        bits = (sc >= 0.5).astype(np.int64)
        g = np.zeros(len(feat_names), dtype=np.int64)
        for j, a in enumerate(vocab):
            g[fidx[f"{a.part}:{a.attr_name}={a.value}"]] = int(gt[j])
        for fi in part_feat_idx:
            p = feat_part[fi]
            visible = vis.get(p, 0.0) > 0.5
            bits[fi] = 1 if visible else 0
            g[fi] = 1 if visible else 0
        return bits, g

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

    tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
    y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
    X_tr = np.stack([bits_and_gt(tr_pooled[s], {p: 1.0 for p in tr_pooled[s]},
                                 tr_gt[s])[0] for s in tr_ids])
    te_ids = sorted([s for s in te_pooled.keys() if sid_to_label_te[s] < K])

    out = {"protocol": "identical to ablation_depth_support.py (seed 42, K=20, oracle boxes)",
           "depths": DEPTHS, "instances": []}
    for depth in DEPTHS:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=4,
                                    random_state=42).fit(X_tr, y_tr)
        fb = FastBinaryTree(dt)
        if depth == DEPTHS[0]:
            for s in te_ids:
                out["instances"].append({"sid": int(s), "true": int(sid_to_label_te[s]),
                                         "by_depth": {}})
        for rec, s in zip(out["instances"], te_ids):
            bits, g = bits_and_gt(te_pooled[s], te_vis[s], te_gt[s])
            pred = int(dt.predict(bits.reshape(1, -1))[0])
            R = reason_of(dt, fb, bits, pred)
            lits = []
            for f, b in R:
                part = feat_part[f]
                visible = te_vis[s].get(part, 0.0) > 0.5
                lits.append({"f": feat_names[f], "b": int(b),
                             "supported": bool(int(g[f]) == b),
                             "part_visible": bool(visible)})
            rec["by_depth"][str(depth)] = {"pred": pred, "correct": pred == rec["true"],
                                           "literals": lits,
                                           "all_supported": bool(all(l["supported"] for l in lits))}
        print(f"depth {depth} dumped", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
