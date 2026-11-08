"""
Script: run_design_b.py
=======================
Design B (2026-11-08): surrogate-explanation audit of a black box
whose "uses a region" is EXACTLY computable.

Step 1  Black box = multinomial linear head on MEAN-POOLED PATCH
        tokens (logit_c = b_c + (1/N) sum_p w_c . t_p), so every
        patch's contribution to the logit is a number.
Step 2  Distill, don't relabel: tree fit on the HEAD's train
        predictions, inputs = the 36 reader bits (annotated boxes).
        Depth sweep; fidelity reported with oracle boxes AND 8px
        jittered boxes (localization caveat made visible).
Step 3  Three numbers per explained instance, thresholds
        pre-registered: (1) reason inside the tau=0.8 trusted-literal
        set; (2) leaf-grouped conditional fidelity P(head = c | leaf)
        with Wilson; (3) region agreement = share of the head's
        POSITIVE logit contribution inside the regions behind R's
        literals, percentile vs 200 random same-size region sets.
        Explained fraction = all three pass: fid >= 0.80,
        agreement percentile >= 95.
Step 4  Causal check: replace patches OUTSIDE R's regions (donor
        image patches; dataset-mean patches) and test head-class
        preservation, vs a random-same-size-region control.

PRE-REGISTERED PREDICTIONS (before the run):
  PB1 overall tree-vs-head fidelity >= 0.75 at some depth at K=4,
     and leaf conditional fidelity spans >= 0.4 across leaves.
  PB2 region agreement above chance (pct >= 95) for localized
     literals (belly-style), ~chance (pct < 90) for bill_color=yellow.
  PB3 explained fraction at K=4 < 25%.
  PB4 interventions: keeping R's regions preserves the head class
     >= random-set preservation + 10 points where region agreement
     is high; ~= random for yellow.
  PB5 8px box jitter costs <= 5 fidelity points.
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
    ap.add_argument("--depths", default="2,3,4,6,8,12")
    ap.add_argument("--jitter-sigma-px", type=float, default=8.0)
    ap.add_argument("--jitter-reps", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--thr-fid", type=float, default=0.80)
    ap.add_argument("--thr-agree-pct", type=float, default=95.0)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
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
    tr_gt = {int(m[0]): m[4] for m in tr_rows}
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

    def vec_from(parts):
        d = {}
        for part, vec in parts.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                with torch.no_grad():
                    t = torch.from_numpy(vec.reshape(1, -1)).float()
                    p = float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
                d[f"{a.part}:{a.attr_name}={a.value}"] = 1 if p >= 0.5 else 0
        for n in part_bits_names:
            d[n] = 1
        return np.array([d.get(n, 0) for n in feat_names], dtype=np.int64)

    tr_vec = {s: vec_from(tr_pooled[s]) for s in sids}

    # ---- trusted literals on the validation split (tau = 0.8) ----------------
    def gt_bits_train(sid):
        bits = {f"{a.part}:{a.attr_name}={a.value}": int(tr_gt[sid][j])
                for j, a in enumerate(vocab)}
        for n in part_bits_names:
            p = n[len("has_part("):-1]
            bits[n] = 1 if p in tr_pooled[sid] else 0
        return bits

    lit_stat = {}
    for sid in val_sids:
        gtb = gt_bits_train(sid)
        for name in feat_names:
            if name in part_bits_names and name not in gtb:
                continue
            bit = int(tr_vec[sid][fidx[name]])
            lit_stat.setdefault((name, bit), [0, 0])
            lit_stat[(name, bit)][1] += 1
            lit_stat[(name, bit)][0] += int(gtb.get(name) == bit)
    trusted = {}
    for (name, pol), (c0, n0) in lit_stat.items():
        if n0 >= 10:
            w = wilson(c0, n0)
            trusted[(name, pol)] = w[0] >= args.tau

    # ---- patch grids ---------------------------------------------------------
    tr_fm = dtr["feature_maps"]            # [N,384,16,16]
    te_fm = dte["feature_maps"]
    tr_order = {int(m[0]): i for i, m in enumerate(tr_rows)}
    te_order = {int(m[0]): i for i, m in enumerate(te_rows)}
    mean_patch = torch.from_numpy(
        tr_fm.astype(np.float32).transpose(0, 2, 3, 1).reshape(-1, 384)).mean(dim=0)
    grid_y = (np.arange(16) + 0.5) / 16.0
    grid_x = (np.arange(16) + 0.5) / 16.0

    def patch_contribs(fm1, cls, Wm, bm):
        """Positive-share contributions per patch: t_p . w_c."""
        toks = torch.from_numpy(fm1.astype(np.float32)).reshape(384, -1).T  # [256,384]
        return (toks @ Wm[cls]).numpy()

    results = []
    for K in classes:
        tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
        te_ids = [s for s in range(len(te_rows)) if sid_to_label_te[te_rows[s][0]] < K]
        # Step 1: patch head
        Xp_tr = np.stack([tr_fm[tr_order[s]].astype(np.float32).mean(axis=(1, 2))
                          for s in tr_ids])
        y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
        head = LogisticRegression(max_iter=3000, C=1.0,
                                  random_state=42).fit(Xp_tr, y_tr)
        Xp_te = np.stack([te_fm[i].astype(np.float32).mean(axis=(1, 2))
                          for i in te_ids])
        y_te = np.array([sid_to_label_te[te_rows[i][0]] for i in te_ids])
        head_te = head.predict(Xp_te)
        head_acc = float(np.mean(head_te == y_te))
        Wm = torch.from_numpy(head.coef_).float()   # [K,384]
        cls_index = {int(c): i for i, c in enumerate(head.classes_)}
        head_tr_pred = head.predict(np.stack(
            [tr_fm[tr_order[s]].astype(np.float32).mean(axis=(1, 2)) for s in tr_ids]))

        # oracle test bits, computed once (depth-independent)
        def oracle_bits(i):
            sid, _m, boxes, vis, _a = te_rows[i]
            parts = {}
            fm1 = torch.from_numpy(te_fm[i].astype(np.float32)).unsqueeze(0)
            for p in ds_train.part_names:
                if vis.get(p, 0.0) > 0.5 and p in boxes:
                    r = Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                    pp = pool_region_features(fm1, [r])
                    parts[p] = pp[r.id].numpy()
            return vec_from(parts), parts
        print(f"--> K={K}: patch head train n={len(tr_ids)} test acc {head_acc:.3f}",
              flush=True)

        # Step 2: distilled trees (oracle + jitter)
        Xb_tr = np.stack([tr_vec[s] for s in tr_ids])
        te_bits, te_parts = zip(*[oracle_bits(i) for i in te_ids])
        Xb_te = np.stack(te_bits)
        sweep = []
        trees = {}
        for D in depths:
            dt = DecisionTreeClassifier(max_depth=D, min_samples_leaf=4,
                                        random_state=42).fit(Xb_tr, head_tr_pred)
            h = CertifiableTreeHead(max_depth=D, min_samples_leaf=4,
                                    feature_names=feat_names)
            h.clf, h.is_fitted = dt, True
            trees[D] = (dt, h, PyXAIBackend(h))
            pred = dt.predict(Xb_te)
            fid = float(np.mean(pred == head_te))
            w = wilson(int((pred == head_te).sum()), len(te_ids))
            # mean reason length
            ks = []
            for j, i in enumerate(te_ids):
                exp = trees[D][2].explain_sufficient(
                    Xb_te[j], int(pred[j]), sample_id=te_rows[i][0])
                ks.append(len(exp.conditions))
            sweep.append({"depth": D, "fidelity_oracle": fid, "wilson95": w,
                          "mean_reason_size": float(np.mean(ks))})
            print(f"    D={D:2d}: fidelity {fid:.3f} {w} | mean k {np.mean(ks):.2f}",
                  flush=True)
        # jittered fidelity at each depth (boxes only, cached maps)
        jit_rows = []
        for D in depths:
            fids = []
            for rep in range(args.jitter_reps):
                rr = np.random.RandomState(2000 + rep)
                ok = 0
                for j, i in enumerate(te_ids):
                    boxes, vis = te_rows[i][2], te_rows[i][3]
                    parts = {}
                    fm1 = torch.from_numpy(te_fm[i].astype(np.float32)).unsqueeze(0)
                    for p in ds_train.part_names:
                        if vis.get(p, 0.0) > 0.5 and p in boxes:
                            y0, x0, y1, x1 = boxes[p]
                            h_, w_ = y1 - y0, x1 - x0
                            cy, cx = (y0 + y1) / 2 + rr.normal(0, args.jitter_sigma_px / 224), \
                                     (x0 + x1) / 2 + rr.normal(0, args.jitter_sigma_px / 224)
                            ny0, nx0 = np.clip(cy - h_ / 2, 0, 1 - h_), np.clip(cx - w_ / 2, 0, 1 - w_)
                            r = Region(id=f"region_{p}", part_name=p,
                                       bbox=(float(ny0), float(nx0),
                                             float(ny0 + h_), float(nx0 + w_)))
                            pp = pool_region_features(fm1, [r])
                            parts[p] = pp[r.id].numpy()
                    v = vec_from(parts)
                    ok += int(trees[D][0].predict(v.reshape(1, -1))[0] == head_te[j])
                fids.append(ok / len(te_ids))
            jit_rows.append({"depth": D,
                             "fidelity_jitter8px": float(np.mean(fids))})
            print(f"    D={D:2d}: jitter {args.jitter_sigma_px:.0f}px fidelity "
                  f"{np.mean(fids):.3f}", flush=True)

        # Step 3: three numbers at D=4 and D=12
        three = {}
        for D in (4, 12):
            dt, h, backend = trees[D]
            rows3 = []
            for j, i in enumerate(te_ids):
                sid, meta_sid, boxes, vis, attrs = te_rows[i]
                v = te_bits[j]
                pred_t = int(dt.predict(v.reshape(1, -1))[0])
                exp = backend.explain_sufficient(v, pred_t, sample_id=sid)
                lits = [(feat_names[c.feature_idx], 1 if c.operator != "<=" else 0)
                        for c in exp.conditions]
                ok_trusted = all(trusted.get(ln, False) for ln, _pol in lits)
                # regions behind R's literals
                regs = []
                for ln, _pol in lits:
                    part = ln[len("has_part("):-1] if ln.startswith("has_part(") \
                        else ln.split(":")[0]
                    if part in boxes and vis.get(part, 0.0) > 0.5:
                        regs.append(boxes[part])
                # leaf + head class
                leaf = int(dt.apply(v.reshape(1, -1))[0])
                c_head = int(head_te[j])
                # region agreement vs null
                pct = obs_share = None
                if regs:
                    ci = cls_index[c_head]
                    contribs = patch_contribs(te_fm[i], ci, Wm, None).reshape(16, 16)
                    pos = np.clip(contribs, 0, None)
                    tot = pos.sum()
                    if tot > 1e-8:
                        rng_null = np.random.RandomState(31337 + j)
                        mask = np.zeros((16, 16), dtype=bool)
                        for (y0, x0, y1, x1) in regs:
                            mask |= ((grid_y[:, None] >= y0) & (grid_y[:, None] <= y1) &
                                     (grid_x[None, :] >= x0) & (grid_x[None, :] <= x1))
                        obs_share = float(pos[mask].sum() / tot)
                        sizes = [(r[2] - r[0], r[3] - r[1]) for r in regs]
                        nul = np.zeros(args.n_null)
                        for t in range(args.n_null):
                            m2 = np.zeros((16, 16), dtype=bool)
                            for (h_, w_) in sizes:
                                y0 = rng_null.uniform(0, 1 - h_)
                                x0 = rng_null.uniform(0, 1 - w_)
                                m2 |= ((grid_y[:, None] >= y0) &
                                       (grid_y[:, None] <= y0 + h_) &
                                       (grid_x[None, :] >= x0) &
                                       (grid_x[None, :] <= x0 + w_))
                            nul[t] = pos[m2].sum() / tot
                        pct = float((nul < obs_share).mean() * 100)
                rows3.append({"sid": sid, "k": len(lits),
                              "literals": [f"{ln}{'+' if p else '-'}" for ln, p in lits],
                              "trusted": ok_trusted, "leaf": leaf,
                              "head_class": c_head, "tree_class": pred_t,
                              "region_share": obs_share, "agree_pct": pct,
                              "n_regions": len(regs)})
            # leaf-grouped conditional fidelity vs the head
            by_leaf = defaultdict(list)
            for r3, hj in zip(rows3, head_te):
                by_leaf[r3["leaf"]].append(int(r3["tree_class"] == hj))
            leaf_fid = {str(lf): {"n": len(v), "fid": float(np.mean(v)),
                                  "wilson95": wilson(sum(v), len(v))}
                        for lf, v in by_leaf.items() if len(v) >= 5}
            explained = sum(
                1 for r3 in rows3
                if r3["trusted"]
                and leaf_fid.get(str(r3["leaf"]), {}).get("fid", 0) >= args.thr_fid
                and r3["agree_pct"] is not None
                and r3["agree_pct"] >= args.thr_agree_pct)
            three[f"D{D}"] = {
                "rows": rows3, "leaf_fidelity": leaf_fid,
                "explained_fraction": explained / len(te_ids),
                "explained_n": explained,
                "fails": {
                    "no_trusted_reason": sum(1 for r3 in rows3 if not r3["trusted"]),
                    "low_leaf_fid": sum(1 for r3 in rows3 if r3["trusted"] and
                                        leaf_fid.get(str(r3["leaf"]), {})
                                        .get("fid", 0) < args.thr_fid),
                    "low_agreement": sum(
                        1 for r3 in rows3 if r3["trusted"] and
                        (r3["agree_pct"] is None or
                         r3["agree_pct"] < args.thr_agree_pct)),
                },
            }
            yellow = [r3 for r3 in rows3
                      if any(l.startswith("beak:bill_color=yellow") for l in r3["literals"])]
            print(f"    D={D}: explained {explained}/{len(te_ids)} "
                  f"({explained/len(te_ids):.3f}) | yellow-literal rows "
                  f"{len(yellow)}, mean agree_pct "
                  f"{np.mean([r3['agree_pct'] for r3 in yellow if r3['agree_pct'] is not None]) if yellow else float('nan'):.0f}",
                  flush=True)

        # Step 4: interventions (D=4)
        D = 4
        dt, h, backend = trees[D]
        rr = np.random.RandomState(777)
        iv = {"donor": {"R": 0, "rand": 0, "n": 0},
              "meanpatch": {"R": 0, "rand": 0, "n": 0}}
        for j, i in enumerate(te_ids):
            sid, meta_sid, boxes, vis, attrs = te_rows[i]
            v = te_bits[j]
            pred_t = int(dt.predict(v.reshape(1, -1))[0])
            exp = backend.explain_sufficient(v, pred_t, sample_id=sid)
            lits = [(feat_names[c.feature_idx], 1 if c.operator != "<=" else 0)
                    for c in exp.conditions]
            regs = []
            for ln, _pol in lits:
                part = ln[len("has_part("):-1] if ln.startswith("has_part(") \
                    else ln.split(":")[0]
                if part in boxes and vis.get(part, 0.0) > 0.5:
                    regs.append(boxes[part])
            if not regs:
                continue
            c0 = int(head_te[j])
            toks = torch.from_numpy(te_fm[i].astype(np.float32)).reshape(384, -1).T
            mask = np.zeros((16, 16), dtype=bool)
            for (y0, x0, y1, x1) in regs:
                mask |= ((grid_y[:, None] >= y0) & (grid_y[:, None] <= y1) &
                         (grid_x[None, :] >= x0) & (grid_x[None, :] <= x1))
            mask_flat = mask.reshape(-1)
            # random same-size region mask control
            sizes = [(r[2] - r[0], r[3] - r[1]) for r in regs]
            m2 = np.zeros((16, 16), dtype=bool)
            for (h_, w_) in sizes:
                y0 = rr.uniform(0, 1 - h_); x0 = rr.uniform(0, 1 - w_)
                m2 |= ((grid_y[:, None] >= y0) & (grid_y[:, None] <= y0 + h_) &
                       (grid_x[None, :] >= x0) & (grid_x[None, :] <= x0 + w_))
            m2_flat = m2.reshape(-1)
            donor_i = te_fm[rr.choice([x for x in range(len(te_rows)) if x != i])]
            donor_toks = torch.from_numpy(donor_i.astype(np.float32)).reshape(384, -1).T
            for scheme, fill in (("donor", donor_toks), ("meanpatch", None)):
                for keep_mask, key in ((mask_flat, "R"), (m2_flat, "rand")):
                    t2 = toks.clone()
                    fillv = fill if fill is not None else mean_patch
                    t2[~torch.from_numpy(keep_mask)] = fillv[
                        torch.from_numpy(~keep_mask)] if fill is not None else fillv
                    x2 = t2.mean(dim=0).numpy().reshape(1, -1)
                    iv[scheme][key] += int(head.predict(x2)[0] == c0)
                iv[scheme]["n"] += 1
        for scheme in iv:
            n = iv[scheme]["n"]
            print(f"    intervention {scheme}: keep-R {iv[scheme]['R']/n:.3f} "
                  f"keep-random {iv[scheme]['rand']/n:.3f} (n={n})", flush=True)

        results.append({
            "K": K, "n_test": len(te_ids), "patch_head_acc": head_acc,
            "depth_sweep": sweep, "jitter_fidelity": jit_rows,
            "three_numbers": three, "interventions": iv,
        })

    report = {
        "protocol": {
            "question": ("Design B: does the distilled tree's certified reason "
                         "explain the MEAN-PATCH-LINEAR head's mechanism, not "
                         "just its output?"),
            "decision": ("black box = multinomial linear head on mean-pooled "
                         "patch tokens (exact per-patch logit decomposition); "
                         "CLS head deferred as the harder case"),
            "thresholds_pre_registered": {
                "trusted_tau": args.tau, "cond_fid": args.thr_fid,
                "agree_percentile": args.thr_agree_pct},
            "predictions_pre_registered": [
                "PB1 fidelity >= 0.75 some depth; leaf fid spans >= 0.4",
                "PB2 agreement pct >= 95 localized, < 90 for yellow",
                "PB3 explained fraction < 25% at K=4",
                "PB4 keep-R preservation >= random + 10pts where agreement high",
                "PB5 jitter 8px costs <= 5 fidelity points",
            ],
            "distillation": "tree labels = head's train predictions (not GT)",
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
