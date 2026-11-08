"""
Script: run_design_b_v2.py
==========================
Design B v2 (2026-11-08) - repairs the two weak measurements and adds
the positive control:

A. FIDELITY CEILING LADDER (which representation is the limit?):
   trees distilled from the patch head's predictions with inputs
     bits36        thresholded reader bits (Design B baseline)
     scores36      continuous reader probabilities (thresholding cost)
     region384     mean of visible parts' pooled region features
                   (vocabulary/pooling cost; no literal reasons)
     gtbits36      ORACLE black box: linear head on the GT annotation
                   bits, distilled from reader bits -- the exact-
                   concept POSITIVE CONTROL for the abstention story
B. MARGIN INTERVENTIONS (continuous, sufficient + necessary):
   margin = logit_c - max_{c'!=c} logit_c'.  Sufficiency: replace the
   complement of R's regions (donor tokens / mean patch) and measure
   margin retention vs a size-matched random-region control.
   Necessity: delete R's regions (replace with fill) and measure the
   margin drop vs random-set deletion.
C. REGION-AGREEMENT SANITY (is the sub-chance percentile a bug?):
   per instance: sign mix (pos-sum / |sum|), near-zero totals,
   corr(contrib, patch token norm), contribution inside part boxes vs
   background (outside all visible part boxes), and the |contrib|
   share as a robustness variant of the metric.

PRE-REGISTERED PREDICTIONS:
  PC1 scores36 fidelity >= bits36 + 5 points; region384 >= 0.75;
     gtbits36 head acc >= 0.85 and distilled fidelity >= 0.80.
  PC2 necessity margin-drop for R's regions <= random-set drop + noise
     (R is not the head's mechanism; consistent with sub-chance
     agreement); sufficiency retention low for both (floor removed by
     continuity, but R ~ random).
  PC3 sanity: positive-share well-defined (pos-sum >= 20% of |sum|)
     for >= 90% of instances; contrib correlates positively with token
     norm; part boxes overlap the bird (center), so sub-chance
     percentiles reflect where the head's mass actually sits, not a
     sign/normalization bug.
  PC4 positive control: gtbits36 arm has trusted-literal coverage > 0
     and explained fraction (criteria 1+2) > 0 at K=4.
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
    ap.add_argument("--k-classes", type=int, default=4)
    ap.add_argument("--depths", default="2,4,6,8,12")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--thr-fid", type=float, default=0.80)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    K = args.k_classes
    depths = [int(d) for d in args.depths.split(",")]
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
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    part_bits_names = [n for n in feat_names if n.startswith("has_part(")]

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

    def probs_and_bits(parts):
        d = {}
        for part, vec in parts.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                with torch.no_grad():
                    t = torch.from_numpy(vec.reshape(1, -1)).float()
                    p = float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
                d[f"{a.part}:{a.attr_name}={a.value}"] = p
        bits = np.array([(1 if d.get(n, 0.0) >= 0.5 else 0) for n in feat_names],
                        dtype=np.int64)
        scores = np.array([d.get(n, 0.0) for n in feat_names], dtype=np.float32)
        return bits, scores

    def region_mean(parts):
        if not parts:
            return np.zeros(384, dtype=np.float32)
        return np.mean(np.stack(list(parts.values())), axis=0)

    def gt_bits_vec(gt, vis):
        bits = {f"{a.part}:{a.attr_name}={a.value}": int(gt[j])
                for j, a in enumerate(vocab)}
        for n in part_bits_names:
            p = n[len("has_part("):-1]
            bits[n] = 1 if vis.get(p, 0.0) > 0.5 else 0
        return np.array([bits.get(n, 0) for n in feat_names], dtype=np.int64)

    tr_variants = {"bits36": {}, "scores36": {}, "region384": {}, "gtbits36": {}}
    for s in sids:
        b, sc = probs_and_bits(tr_pooled[s])
        tr_variants["bits36"][s] = b
        tr_variants["scores36"][s] = sc
        tr_variants["region384"][s] = region_mean(tr_pooled[s])
        vis_tr = {p: 1.0 for p in tr_pooled[s]}
        tr_variants["gtbits36"][s] = gt_bits_vec(tr_gt[s], vis_tr)

    # trusted literals on validation split (tau)
    lit_stat = {}
    for sid in val_sids:
        gtb = gt_bits_vec(tr_gt[sid], {p: 1.0 for p in tr_pooled[sid]})
        for name in feat_names:
            bit = int(tr_variants["bits36"][sid][fidx[name]])
            lit_stat.setdefault((name, bit), [0, 0])
            lit_stat[(name, bit)][1] += 1
            lit_stat[(name, bit)][0] += int(gtb[fidx[name]] == bit)
    trusted = {}
    for (name, pol), (c0, n0) in lit_stat.items():
        if n0 >= 10:
            trusted[(name, pol)] = wilson(c0, n0)[0] >= args.tau

    tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
    te_ids = [i for i in range(len(te_rows)) if sid_to_label_te[te_rows[i][0]] < K]
    te_order = {int(m[0]): i for i, m in enumerate(te_rows)}
    tr_order = {int(m[0]): i for i, m in enumerate(tr_rows)}
    tr_fm, te_fm = dtr["feature_maps"], dte["feature_maps"]

    # ---- black boxes ---------------------------------------------------------
    Xp_tr = np.stack([tr_fm[tr_order[s]].astype(np.float32).mean(axis=(1, 2))
                      for s in tr_ids])
    y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
    patch_head = LogisticRegression(max_iter=3000, C=1.0,
                                    random_state=42).fit(Xp_tr, y_tr)
    Xp_te = np.stack([te_fm[i].astype(np.float32).mean(axis=(1, 2)) for i in te_ids])
    y_te = np.array([sid_to_label_te[te_rows[i][0]] for i in te_ids])
    patch_head_te = patch_head.predict(Xp_te)
    patch_head_acc = float(np.mean(patch_head_te == y_te))

    Xg_tr = np.stack([tr_variants["gtbits36"][s] for s in tr_ids])
    gt_head = LogisticRegression(max_iter=3000, C=1.0,
                                 random_state=42).fit(Xg_tr, y_tr)
    Xg_te = np.stack([gt_bits_vec(te_gt[te_rows[i][0]], te_rows[i][3]) for i in te_ids])
    gt_head_te = gt_head.predict(Xg_te)
    gt_head_acc = float(np.mean(gt_head_te == y_te))
    print(f"--> patch head acc {patch_head_acc:.3f} | GT-bits head acc {gt_head_acc:.3f}",
          flush=True)

    # ---- A. ceiling ladder ----------------------------------------------------
    ladder = {}
    for vname in ("bits36", "scores36", "region384", "gtbits36"):
        X_tr_v = np.stack([tr_variants[vname][s] for s in tr_ids])
        if vname == "gtbits36":
            labels = gt_head.predict(X_tr_v)
            head_te_v = gt_head_te
        else:
            labels = patch_head.predict(Xp_tr)
            head_te_v = patch_head_te
        X_te_v = np.stack([
            (lambda r: probs_and_bits(r)[0] if vname == "bits36" else
             probs_and_bits(r)[1] if vname == "scores36" else
             region_mean(r) if vname == "region384" else
             gt_bits_vec(te_gt[te_rows[i][0]], te_rows[i][3]))(te_pooled[te_rows[i][0]])
            for i in te_ids])
        rows = []
        for D in depths:
            dt = DecisionTreeClassifier(max_depth=D, min_samples_leaf=4,
                                        random_state=42).fit(X_tr_v, labels)
            pred = dt.predict(X_te_v)
            hit = int((pred == head_te_v).sum())
            rows.append({"depth": D, "fidelity": hit / len(te_ids),
                         "wilson95": wilson(hit, len(te_ids))})
            print(f"    {vname:10s} D={D:2d}: fidelity {hit/len(te_ids):.3f}",
                  flush=True)
        ladder[vname] = {"rows": rows,
                         "black_box": "gt_head" if vname == "gtbits36" else "patch_head"}

    # ---- distilled D=4 tree on bits (for B and the abstention control) -------
    Xb_tr = np.stack([tr_variants["bits36"][s] for s in tr_ids])
    dt_patch = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4,
                                      random_state=42).fit(
                                          Xb_tr, patch_head.predict(Xp_tr))
    h = CertifiableTreeHead(max_depth=4, min_samples_leaf=4, feature_names=feat_names)
    h.clf, h.is_fitted = dt_patch, True
    backend = PyXAIBackend(h)

    dt_gt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4,
                                   random_state=42).fit(
                                       Xb_tr, gt_head.predict(Xg_tr))
    hg = CertifiableTreeHead(max_depth=4, min_samples_leaf=4, feature_names=feat_names)
    hg.clf, hg.is_fitted = dt_gt, True
    backend_g = PyXAIBackend(hg)

    # ---- B. margin interventions + C. sanity ---------------------------------
    Wm = torch.from_numpy(patch_head.coef_).float()
    cls_index = {int(c): i for i, c in enumerate(patch_head.classes_)}
    mean_patch = torch.from_numpy(
        tr_fm.astype(np.float32).transpose(0, 2, 3, 1).reshape(-1, 384)).mean(dim=0)
    grid_y = (np.arange(16) + 0.5) / 16.0
    grid_x = (np.arange(16) + 0.5) / 16.0
    rr = np.random.RandomState(777)

    def logits(toks):
        z = toks.mean(dim=0) @ Wm.T + torch.from_numpy(
            patch_head.intercept_.astype(np.float32))
        return z.numpy()

    def margin(z, c):
        ci = cls_index[c]
        others = np.delete(z, ci)
        return float(z[ci] - others.max())

    margin_rows = {"sufficiency": defaultdict(list), "necessity": defaultdict(list)}
    sanity_rows = []
    abstain_ctrl = {"patch": {"cov": 0, "expl": 0}, "gt": {"cov": 0, "expl": 0}}
    leaf_fid_patch, leaf_fid_gt = defaultdict(list), defaultdict(list)
    for j, i in enumerate(te_ids):
        sid, meta_sid, boxes, vis, attrs = te_rows[i]
        parts = te_pooled[sid]
        v, _sc = probs_and_bits(parts)
        fm1 = te_fm[i].astype(np.float32)
        toks = torch.from_numpy(fm1).reshape(384, -1).T
        c = int(patch_head_te[j])

        exp = backend.explain_sufficient(v, int(dt_patch.predict(v.reshape(1, -1))[0]),
                                         sample_id=sid)
        lits = [(feat_names[cc.feature_idx], 1 if cc.operator != "<=" else 0)
                for cc in exp.conditions]
        regs = []
        for ln, _pol in lits:
            part = ln[len("has_part("):-1] if ln.startswith("has_part(") \
                else ln.split(":")[0]
            if part in boxes and vis.get(part, 0.0) > 0.5:
                regs.append(boxes[part])
        # abstention control (patch arm)
        if all(trusted.get(ln, False) for ln, _ in lits):
            abstain_ctrl["patch"]["cov"] += 1
        leaf_fid_patch[int(dt_patch.apply(v.reshape(1, -1))[0])].append(
            int(int(dt_patch.predict(v.reshape(1, -1))[0]) == c))
        # gt arm
        expg = backend_g.explain_sufficient(v, int(dt_gt.predict(v.reshape(1, -1))[0]),
                                            sample_id=sid)
        litsg = [(feat_names[cc.feature_idx], 1 if cc.operator != "<=" else 0)
                 for cc in expg.conditions]
        if all(trusted.get(ln, False) for ln, _pol in litsg):
            abstain_ctrl["gt"]["cov"] += 1
        leaf_fid_gt[int(dt_gt.apply(v.reshape(1, -1))[0])].append(
            int(int(dt_gt.predict(v.reshape(1, -1))[0]) == int(gt_head_te[j])))

        # sanity: contribution structure
        ci = cls_index[c]
        contrib = (toks @ Wm[ci]).numpy()
        pos_sum, abs_sum, tot_sum = contrib[contrib > 0].sum(), np.abs(contrib).sum(), contrib.sum()
        token_norm = toks.norm(dim=1).numpy()
        corr = float(np.corrcoef(contrib, token_norm)[0, 1]) if token_norm.std() > 0 else None
        mask_parts = np.zeros((16, 16), dtype=bool)
        for p in ds_train.part_names:
            if vis.get(p, 0.0) > 0.5 and p in boxes:
                y0, x0, y1, x1 = boxes[p]
                mask_parts |= ((grid_y[:, None] >= y0) & (grid_y[:, None] <= y1) &
                               (grid_x[None, :] >= x0) & (grid_x[None, :] <= x1))
        cm = contrib.reshape(16, 16)
        sanity_rows.append({
            "pos_over_abs": float(pos_sum / abs_sum) if abs_sum > 0 else None,
            "well_defined": bool(pos_sum >= 0.2 * abs(abs_sum) and pos_sum > 0),
            "corr_contrib_norm": corr,
            "mean_contrib_in_parts": float(cm[mask_parts].mean()) if mask_parts.any() else None,
            "mean_contrib_bg": float(cm[~mask_parts].mean()),
        })

        if not regs:
            continue
        mask = np.zeros((16, 16), dtype=bool)
        for (y0, x0, y1, x1) in regs:
            mask |= ((grid_y[:, None] >= y0) & (grid_y[:, None] <= y1) &
                     (grid_x[None, :] >= x0) & (grid_x[None, :] <= x1))
        sizes = [(r[2] - r[0], r[3] - r[1]) for r in regs]
        m2 = np.zeros((16, 16), dtype=bool)
        for (h_, w_) in sizes:
            y0 = rr.uniform(0, 1 - h_); x0 = rr.uniform(0, 1 - w_)
            m2 |= ((grid_y[:, None] >= y0) & (grid_y[:, None] <= y0 + h_) &
                   (grid_x[None, :] >= x0) & (grid_x[None, :] <= x0 + w_))
        donor_toks = torch.from_numpy(
            te_fm[rr.choice([x for x in range(len(te_rows)) if x != i])]
            .astype(np.float32)).reshape(384, -1).T
        base = margin(logits(toks), c)
        for scheme, fill in (("donor", donor_toks), ("meanpatch", None)):
            for keep_mask, tag in ((mask, "R"), (m2, "rand")):
                km = torch.from_numpy(keep_mask.reshape(-1))
                # sufficiency: keep keep_mask, replace complement
                t2 = toks.clone()
                fillv = fill if fill is not None else mean_patch
                t2[~km] = fillv[~km] if fill is not None else fillv
                margin_rows["sufficiency"][f"{scheme}|{tag}"].append(
                    margin(logits(t2), c) - base)
                # necessity: delete keep_mask (replace it)
                t3 = toks.clone()
                t3[km] = fillv[km] if fill is not None else fillv
                margin_rows["necessity"][f"{scheme}|{tag}"].append(
                    margin(logits(t3), c) - base)

    # explained fraction for the gt arm (criteria 1+2; region n/a for a
    # concept-space black box)
    leaf_gt_fid = {lf: float(np.mean(v)) for lf, v in leaf_fid_gt.items() if len(v) >= 5}
    # recompute per-instance gt explained
    gt_expl = 0
    for j, i in enumerate(te_ids):
        sid = te_rows[i][0]
        v, _sc = probs_and_bits(te_pooled[sid])
        expg = backend_g.explain_sufficient(v, int(dt_gt.predict(v.reshape(1, -1))[0]),
                                            sample_id=sid)
        litsg = [(feat_names[cc.feature_idx], 1 if cc.operator != "<=" else 0)
                 for cc in expg.conditions]
        lf = int(dt_gt.apply(v.reshape(1, -1))[0])
        if (all(trusted.get(ln, False) for ln, _ in litsg)
                and leaf_gt_fid.get(lf, 0) >= args.thr_fid):
            gt_expl += 1
    abstain_ctrl["gt"]["expl"] = gt_expl
    leaf_patch_fid = {lf: float(np.mean(v)) for lf, v in leaf_fid_patch.items()
                      if len(v) >= 5}
    patch_expl = 0
    for j, i in enumerate(te_ids):
        sid = te_rows[i][0]
        v, _sc = probs_and_bits(te_pooled[sid])
        exp = backend.explain_sufficient(v, int(dt_patch.predict(v.reshape(1, -1))[0]),
                                         sample_id=sid)
        lits = [(feat_names[cc.feature_idx], 1 if cc.operator != "<=" else 0)
                for cc in exp.conditions]
        lf = int(dt_patch.apply(v.reshape(1, -1))[0])
        if (all(trusted.get(ln, False) for ln, _ in lits)
                and leaf_patch_fid.get(lf, 0) >= args.thr_fid):
            patch_expl += 1
    abstain_ctrl["patch"]["expl"] = patch_expl

    report = {
        "protocol": {
            "question": ("Design B v2: ceiling ladder, margin interventions, "
                         "agreement sanity, GT-concept positive control"),
            "predictions_pre_registered": ["PC1", "PC2", "PC3", "PC4"],
            "note": ("region agreement is undefined for the gtbits36 arm (black box "
                     "lives in concept space); explained fraction there = criteria "
                     "1+2 only"),
        },
        "heads": {"patch_head_acc": patch_head_acc, "gt_head_acc": gt_head_acc},
        "ladder": ladder,
        "margins": {m: {k: {"mean": float(np.mean(v)), "n": len(v)}
                        for k, v in tags.items()}
                    for m, tags in margin_rows.items()},
        "sanity": {
            "well_defined_rate": float(np.mean([r["well_defined"] for r in sanity_rows])),
            "mean_pos_over_abs": float(np.mean([r["pos_over_abs"] for r in sanity_rows
                                                if r["pos_over_abs"] is not None])),
            "mean_corr_contrib_norm": float(np.mean([r["corr_contrib_norm"]
                                                     for r in sanity_rows
                                                     if r["corr_contrib_norm"] is not None])),
            "mean_contrib_in_parts": float(np.mean([r["mean_contrib_in_parts"]
                                                    for r in sanity_rows
                                                    if r["mean_contrib_in_parts"] is not None])),
            "mean_contrib_bg": float(np.mean([r["mean_contrib_bg"] for r in sanity_rows])),
        },
        "abstention_control": abstain_ctrl,
        "leaf_fidelity_patch": {str(k): round(v, 3) for k, v in leaf_patch_fid.items()},
        "leaf_fidelity_gt": {str(k): round(v, 3) for k, v in leaf_gt_fid.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n--> margins (mean delta):")
    for m, tags in margin_rows.items():
        print(f"    {m}: " + ", ".join(f"{k} {np.mean(v):+.2f}" for k, v in tags.items()))
    print(f"--> sanity: well-defined {report['sanity']['well_defined_rate']:.2f} | "
          f"pos/abs {report['sanity']['mean_pos_over_abs']:.2f} | "
          f"corr(contrib,norm) {report['sanity']['mean_corr_contrib_norm']:.2f} | "
          f"contrib parts {report['sanity']['mean_contrib_in_parts']:+.3f} vs "
          f"bg {report['sanity']['mean_contrib_bg']:+.3f}")
    print(f"--> abstention control: patch cov {abstain_ctrl['patch']['cov']}/118 "
          f"expl {abstain_ctrl['patch']['expl']} | gt cov "
          f"{abstain_ctrl['gt']['cov']}/118 expl {abstain_ctrl['gt']['expl']}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
