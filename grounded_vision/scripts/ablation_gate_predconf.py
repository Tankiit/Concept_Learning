#!/usr/bin/env python3
"""Gate ablation: prediction-confidence baseline vs per-literal channels.

Question (reviewer): "abstention here is just selective prediction --
would abstaining on the PREDICTION's confidence do the same job?"
Channels compared at matched coverage, K=10, Setting T (task3 protocol,
seed 42, exactly reproducible):
  B   min calibrated polarity-aware p_true over reason literals (ours)
  E   max class probability of t(phi(x))  (== leaf purity for a CART;
      the standard selective-prediction score; same value for every
      literal of the instance)
  C   min raw reader confidence
  A   oracle in-reason marginal support
Per instance: greedy-irreducible path reason, per-literal support,
calibrated p_true (val-carve isotonic), tree proba. Coverage sweep with
cluster-bootstrap LB of the support ratio.

Output: results/ablation_gate_predconf.json (refuses to overwrite).
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from grounded_vision.data.cub import CUBDataset  # noqa: E402
from grounded_vision.perception.regions import Region, pool_region_features  # noqa: E402
from grounded_vision.perception.readers import AttributeReader  # noqa: E402
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode  # noqa: E402
from run_cbm_cem_audit import FastBinaryTree  # noqa: E402

CACHE = "/Users/tanmoy/research/data/grounded_vision_runs/cub_local_et"
DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
OUT = os.path.join(REPO, "results", "ablation_gate_predconf.json")
K = 10
COVERAGES = [0.05, 0.10, 0.136, 0.20, 0.30]


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

    def scores_bits_gt(parts, vis, gt):
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
            bits[fi] = 1 if vis.get(p, 0.0) > 0.5 else 0
            g[fi] = 1 if vis.get(p, 0.0) > 0.5 else 0
        return sc, bits, g

    # val isotonic per feature (polarity-aware calibration source)
    val_pairs = defaultdict(list)
    for sid in val_sids:
        sc, _, g = scores_bits_gt(tr_pooled[sid], {q: 1.0 for q in tr_pooled[sid]}, tr_gt[sid])
        for name, fi in fidx.items():
            if not name.startswith("has_part("):
                val_pairs[fi].append((sc[fi], float(g[fi])))
    iso = {}
    for fi, pairs in val_pairs.items():
        if len(pairs) < 10:
            continue
        xs, ys = zip(*pairs)
        if min(ys) == max(ys):
            iso[fi] = ("const", float(ys[0]))
        else:
            iso[fi] = ("ir", IsotonicRegression(out_of_bounds="clip").fit(xs, ys))

    def cal(fi, s):
        if fi not in iso:
            return s
        kind, obj = iso[fi]
        return float(obj) if kind == "const" else float(obj.predict([s])[0])

    tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
    te_ids = sorted([s for s in te_pooled.keys() if sid_to_label_te[s] < K])
    y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
    X_tr = np.stack([scores_bits_gt(tr_pooled[s], {p: 1.0 for p in tr_pooled[s]},
                                    tr_gt[s])[1] for s in tr_ids])
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                random_state=42).fit(X_tr, y_tr)
    fb = FastBinaryTree(dt)

    def reason_of(bits, pred):
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

    rows = []
    for s in te_ids:
        sc, bits, g = scores_bits_gt(te_pooled[s], te_vis[s], te_gt[s])
        pred = int(dt.predict(bits.reshape(1, -1))[0])
        proba = float(np.max(dt.predict_proba(bits.reshape(1, -1))[0]))
        R = reason_of(bits, pred)
        if not R:
            continue
        lits, sup = [], []
        for f, b in R:
            lits.append((f, b))
            sup.append(bool(int(g[f]) == b))
        rows.append({
            "sid": int(s),
            "supported": bool(all(sup)),
            "B": float(min((cal(f, sc[f]) if b == 1 else 1 - cal(f, sc[f]))
                           for f, b in lits)),
            "C": float(min((sc[f] if b == 1 else 1 - sc[f]) for f, b in lits)),
            "E": proba,
        })
    # oracle marginal channel
    occ = defaultdict(list)
    for s in te_ids:
        sc, bits, g = scores_bits_gt(te_pooled[s], te_vis[s], te_gt[s])
        pred = int(dt.predict(bits.reshape(1, -1))[0])
        for f, b in reason_of(bits, pred):
            occ[(f, b)].append(bool(int(g[f]) == b))
    marg = {k: float(np.mean(v)) for k, v in occ.items()}
    for r in rows:
        sc, bits, g = scores_bits_gt(te_pooled[r["sid"]], te_vis[r["sid"]], te_gt[r["sid"]])
        pred = int(dt.predict(bits.reshape(1, -1))[0])
        r["A"] = float(min(marg[(f, b)] for f, b in reason_of(bits, pred)))

    y = np.array([r["supported"] for r in rows], float)
    n = len(rows)
    R = float(y.mean())
    print(f"n={n}, unconditional support R={R:.3f} (committed T/K=10: 0.115)")

    boot = np.random.default_rng(42)
    out = {"protocol": "task3-protocol K=10 Setting T; E = max tree class prob "
                       "(leaf purity); channels B/E/C/A per instance",
           "n": n, "unconditional_R": R,
           "spearman_B_E": float(np.corrcoef(np.argsort(np.argsort([r['B'] for r in rows])),
                                             np.argsort(np.argsort([r['E'] for r in rows])))[0, 1]),
           "at_coverage": {}}
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        cell = {}
        for ch in ("B", "E", "C", "A"):
            s_ = np.array([r[ch] for r in rows])
            v = float(y[np.argsort(-s_)[:k]].mean())
            ratios = []
            for _ in range(10000):
                idx = boot.integers(0, n, n)
                kk = max(1, int(round(c * n)))
                ratios.append(y[idx][np.argsort(-s_[idx])[:kk]].mean() / max(1e-9, y[idx].mean()))
            cell[ch] = {"support": round(v, 4),
                        "LB_ratio": round(float(np.percentile(ratios, 2.5)), 2)}
        out["at_coverage"][f"{c:.3f}"] = cell
        print(f"cov {c:.3f}: " + "  ".join(
            f"{ch}={cell[ch]['support']:.3f}(LB {cell[ch]['LB_ratio']})" for ch in ("B", "E", "C", "A")))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
