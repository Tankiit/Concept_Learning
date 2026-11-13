"""
bank.py — DINO tokens -> region predicates -> binary concept bank.
====================================================================
Stage 1 of the pipeline (PyC env: conda torch-multimodal).

Frozen committed protocol, unchanged from run_cbm_cem_audit.py /
run_guarantee_scaling_cub.py: cached DINOv3 feature maps + oracle GT
part boxes; logistic concept readers on pooled region features trained
on the 85% fit split (15% validation carve-out for isotonic literal
calibrators); bits thresholded at 0.5; has_part features structural
(bit 1, Setting-T convention); unread (unobserved-part) concepts get
score 0 / bit 0.

Fit once (K-agnostic); downstream stages filter by class count.
Reuses the cached bank unless --rebuild is passed.

Usage
-----
    python src/bank.py --cache-dir <dino cache> --data-dir <CUB root>

Outputs (intermediate_results/, not tracked)
    bank_train.npz / bank_test.npz
        sample_ids (int64), labels (int64), bits (int8 n x F),
        scores (float32), cal (float32), gt (int8 annotation bits)
    meta.json
        feature_names, feat_part, part_feat_idx, vocab, val_sids,
        concept_quality, protocol block
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_SRC = PROJECT_ROOT.parent / "src"
sys.path.insert(0, str(REPO_SRC))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"


def to_binary(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Continuous concept scores -> bits (unread concepts must already
    be encoded as 0-score / absent)."""
    return (np.asarray(scores) >= threshold).astype(np.int8)


def fit_predicate_bank(cache_dir, data_dir, out_dir=INTERMEDIATE_DIR, seed=42,
                       epochs=40, lr=1e-3, val_fraction=0.15):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtr = np.load(Path(cache_dir) / "train_features.npz")
    dte = np.load(Path(cache_dir) / "test_features.npz")
    ds_train = CUBDataset(data_dir=data_dir, split="train",
                          use_mock_if_missing=False, seed=42)
    ds_test = CUBDataset(data_dir=data_dir, split="test",
                         use_mock_if_missing=False, seed=42)
    vocab = ds_train.bound_attributes
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=list(vocab))
    feat_names = compiler.feature_names
    fidx = compiler.feature_to_idx
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    part_feat_idx = [fidx[n] for n in feat_names if n.startswith("has_part(")]
    feat_part = []
    for n in feat_names:
        feat_part.append(n[len("has_part("):-1] if n.startswith("has_part(")
                         else n.split(":", 1)[0])

    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])

    def pooled(rows, fmaps):
        res = {}
        fm = torch.from_numpy(fmaps.astype(np.float32))
        for i, row in enumerate(rows):
            sid, _m, boxes, vis, _a = row
            regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                    for p in ds_train.part_names
                    if vis.get(p, 0.0) > 0.5 and p in boxes]
            if regs:
                p = pool_region_features(fm[i:i + 1], regs)
                res[int(sid)] = {r.part_name: p[r.id].numpy() for r in regs}
        return res

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    print(f"--> pooling train {len(tr_rows)} / test {len(te_rows)}", flush=True)
    tr_pooled = pooled(tr_rows, dtr["feature_maps"])
    te_pooled = pooled(te_rows, dte["feature_maps"])

    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    te_gt = {int(m[0]): m[4] for m in te_rows}
    te_vis = {int(m[0]): m[3] for m in te_rows}
    tr_vis = {int(m[0]): m[3] for m in tr_rows}

    sids = np.array(sorted(tr_pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(val_fraction * len(sids))
    val_sids = set(sids[perm[:n_val]].tolist())
    fit_sids = set(sids[perm[n_val:]].tolist())

    reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
    opt = torch.optim.Adam(reader.parameters(), lr=lr)
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
    for _ in range(epochs):
        loss = 0.0
        for X, y, pw, cols in region_data.values():
            W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
            b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
            loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"--> readers trained (fit {len(fit_sids)} / val {len(val_sids)})",
          flush=True)

    def scores_map(parts):
        d = {}
        for part, vec in parts.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                t = torch.from_numpy(vec.reshape(1, -1)).float()
                with torch.no_grad():
                    d[f"{a.part}:{a.attr_name}={a.value}"] = float(
                        torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
        return d

    val_scores = defaultdict(list)
    for sid in val_sids:
        d = scores_map(tr_pooled[sid])
        for name, sc in d.items():
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

    def vecs(parts, vis):
        d = scores_map(parts)
        sc = np.zeros(len(feat_names), dtype=np.float64)
        scc = np.zeros(len(feat_names), dtype=np.float64)
        for name, p in d.items():
            fi = fidx[name]
            sc[fi] = p
            scc[fi] = cal_prob(fi, p)
        for fi in part_feat_idx:
            bit = 1.0 if vis.get(feat_part[fi], 0.0) > 0.5 else 0.0
            sc[fi] = bit
            scc[fi] = bit
        return to_binary(sc), sc, scc

    def gt_vec(gt, vis):
        bits = np.zeros(len(feat_names), dtype=np.int8)
        for j, a in enumerate(vocab):
            bits[fidx[f"{a.part}:{a.attr_name}={a.value}"]] = int(gt[j])
        for n in feat_names:
            if n.startswith("has_part("):
                p = n[len("has_part("):-1]
                bits[fidx[n]] = 1 if vis.get(p, 0.0) > 0.5 else 0
        return bits

    def build(rows, pooled, gt_map, vis_map, label_map):
        ids = [int(m[0]) for m in rows if int(m[0]) in pooled]
        VB, VS, VC, VG, VY = [], [], [], [], []
        for s in ids:
            b_, s_, c_ = vecs(pooled[s], vis_map[s])
            VB.append(b_); VS.append(s_); VC.append(c_)
            VG.append(gt_vec(gt_map[s], vis_map[s]))
            VY.append(label_map[s])
        return (np.array(ids, dtype=np.int64), np.array(VY, dtype=np.int64),
                np.stack(VB).astype(np.int8), np.stack(VS).astype(np.float32),
                np.stack(VC).astype(np.float32), np.stack(VG).astype(np.int8))

    lab_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    lab_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}
    tr_pack = build(tr_rows, tr_pooled, tr_gt, tr_vis, lab_tr)
    te_pack = build(te_rows, te_pooled, te_gt, te_vis, lab_te)
    np.savez(out / "bank_train.npz", sample_ids=tr_pack[0], labels=tr_pack[1],
             bits=tr_pack[2], scores=tr_pack[3], cal=tr_pack[4], gt=tr_pack[5])
    np.savez(out / "bank_test.npz", sample_ids=te_pack[0], labels=te_pack[1],
             bits=te_pack[2], scores=te_pack[3], cal=te_pack[4], gt=te_pack[5])

    concept_quality = {}
    for j, a in enumerate(vocab):
        name = f"{a.part}:{a.attr_name}={a.value}"
        fi = fidx[name]
        ok = tot = 0
        for i, sid in enumerate(te_pack[0]):
            if a.part in te_pooled[int(sid)]:
                ok += int(int(te_pack[2][i][fi]) == int(te_pack[5][i][fi]))
                tot += 1
        concept_quality[name] = {"n_observed": tot,
                                 "accuracy": round(ok / tot, 4) if tot else None}
    meta = {
        "feature_names": feat_names,
        "feat_part": feat_part,
        "part_feat_idx": part_feat_idx,
        "vocab": [[a.part, a.attr_name, a.value] for a in vocab],
        "val_sids": sorted(val_sids),
        "concept_quality": concept_quality,
        "protocol": {
            "cache": str(Path(cache_dir)), "seed": seed, "epochs": epochs,
            "localisation": "oracle GT part boxes (labelled limitation)",
            "convention": ("has_part structural bit 1 (Setting T); unread "
                           "concepts score 0 / bit 0; cal = val-isotonic "
                           "P(gt bit | score)"),
            "n_train": int(len(tr_pack[0])), "n_test": int(len(te_pack[0])),
            "n_features": len(feat_names),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    accs = [v["accuracy"] for v in concept_quality.values() if v["accuracy"]]
    print(f"--> bank saved to {out}: F={len(feat_names)} "
          f"train={len(tr_pack[0])} test={len(te_pack[0])} "
          f"concept-acc mean {np.mean(accs):.4f}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--rebuild", action="store_true",
                    help="refit even if bank_train.npz exists")
    args = ap.parse_args()
    if not args.rebuild and (Path(args.out) / "bank_train.npz").exists():
        print(f"--> bank already present at {args.out}; pass --rebuild to refit")
        return
    fit_predicate_bank(args.cache_dir, args.data_dir, args.out,
                       seed=args.seed, epochs=args.epochs)


if __name__ == "__main__":
    main()
