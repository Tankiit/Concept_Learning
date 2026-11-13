"""
Script: run_localization_noise.py
==================================
Round 7 (2026-11-08): localization-noise test.  The image is UNCHANGED
(cached feature maps reused); the fixed part-box CENTERS are jittered
by Gaussian noise (sigma in px at 224 res).  This measures how the
certified guarantee degrades with region-assignment error -- the
deployment-without-annotations scenario, where boxes come from a
detector with some center-error distribution.

Per (sigma, rep): re-pool cached maps at jittered boxes -> readers ->
bits -> tree prediction + minimal reason.  Metrics vs the unjittered
baseline: bit-flip rate (overall and per part family), prediction /
reason stability (k=1 vs k>1), all-clauses-correct rate, the certified
mechanism cell ({bill_color=yellow = 0} single clause), and
conditional fidelity against the linear reference head (logreg trained
once on base train bits), including the certified pattern's fidelity
on jittered bits.

Detector arm (pending): a --detector-checkpoint that supplies
predicted boxes would be evaluated by the same code path and placed on
the jitter curve by its measured center-error sigma.  The only local
checkpoint (cache/cub_bound) is the MOCK smoke run -- unusable.

PRE-REGISTERED PREDICTIONS (before the run):
  PN1 bit-flip rate < 8% at 4px, 10-30% at 16px.
  PN2 k=1 reason stability >= 0.9 at 4px and 8px.
  PN3 k>1 reason stability < k=1 stability at every sigma.
  PN4 certified-pattern conditional fidelity >= 0.9 through 8px.
  PN5 all-correct within 5 points of 25.4% through 8px; drops at 16px.
  PN6 beak concepts flip more than belly concepts (smaller box).
"""

from __future__ import annotations
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigmas", default="0,4,8,16")
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
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
    part_bits = [n for n in feat_names if n.startswith("has_part(")]

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
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

    sids = np.array(sorted(tr_pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(args.val_fraction * len(sids))
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
        for n in part_bits:
            d[n] = 1
        return np.array([d.get(n, 0) for n in feat_names], dtype=np.int64)

    tr_vec = {s: vec_from(tr_pooled[s]) for s in sorted(fit_sids)}

    # K=4 tree + linear reference
    tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < 4]
    X_fit = np.stack([tr_vec[s] for s in tr_ids])
    y_fit = np.array([sid_to_label_tr[s] for s in tr_ids])
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                random_state=42).fit(X_fit, y_fit)
    head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                               feature_names=feat_names)
    head.clf, head.is_fitted = dt, True
    backend = PyXAIBackend(head)
    lr_model = LogisticRegression(max_iter=2000, C=1.0,
                                  random_state=42).fit(X_fit, y_fit)

    K4_rows = [r for r in te_rows if sid_to_label_te[r[0]] < 4]
    te_fm = torch.from_numpy(dte["feature_maps"].astype(np.float32))
    te_order = {int(r[0]): i for i, r in enumerate(te_rows)}
    yj = fidx["beak:bill_color=yellow"]

    def jitter_boxes(boxes, sigma_n, rep_rng):
        out = {}
        for p, (y0, x0, y1, x1) in boxes.items():
            h, w = y1 - y0, x1 - x0
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            cy += rep_rng.normal(0, sigma_n)
            cx += rep_rng.normal(0, sigma_n)
            ny0, nx0 = np.clip(cy - h / 2, 0, 1 - h), np.clip(cx - w / 2, 0, 1 - w)
            out[p] = (float(ny0), float(nx0), float(ny0 + h), float(nx0 + w))
        return out

    def bits_with_boxes(boxes, vis, fm1):
        parts = {}
        for p in ds_train.part_names:
            if vis.get(p, 0.0) > 0.5 and p in boxes:
                r = Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                pp = pool_region_features(fm1, [r])
                parts[p] = pp[r.id].numpy()
        return vec_from(parts), parts

    # baseline (sigma 0, must equal cache bits)
    base = {}
    for sid, meta_sid, boxes, vis, attrs in K4_rows:
        v, _ = bits_with_boxes(boxes, vis, te_fm[te_order[sid]:te_order[sid] + 1])
        gtb = {f"{a.part}:{a.attr_name}={a.value}": int(attrs[j])
               for j, a in enumerate(vocab)}
        pred = int(dt.predict(v.reshape(1, -1))[0])
        exp = backend.explain_sufficient(v, pred, sample_id=sid)
        lits = tuple(sorted((int(c.feature_idx), 1 if c.operator != "<=" else 0)
                            for c in exp.conditions))
        base[sid] = {"v": v, "pred": pred, "lits": lits,
                     "allc": all(gtb.get(feat_names[f]) == w for f, w in lits),
                     "k1": (len(lits) == 1 and lits[0] == (yj, 0))}
    n = len(K4_rows)
    base_allc = float(np.mean([b["allc"] for b in base.values()]))
    print(f"--> baseline: acc-vs-tree-consistent n={n}, allc {base_allc:.3f}", flush=True)

    rows_out = []
    for sigma in sigmas:
        reps = []
        for rep in range(args.n_rep if sigma > 0 else 1):
            rr = np.random.RandomState(1000 + rep)
            flip = flip_beak = flip_belly = 0
            nread = nbeak = nbelly = 0
            pred_same = reason_same = k1_same = kgt_same = allc = 0
            n_k1 = n_kgt = 0
            agree = 0
            cert_mask_n = cert_mask_k = 0
            for sid, meta_sid, boxes, vis, attrs in K4_rows:
                b = base[sid]
                jb = boxes if sigma == 0 else jitter_boxes(boxes, sigma / 224.0, rr)
                v, _ = bits_with_boxes(jb, vis, te_fm[te_order[sid]:te_order[sid] + 1])
                read_idx = [j for j, nm in enumerate(feat_names)
                            if not nm.startswith("has_part")]
                flip += int((v[read_idx] != b["v"][read_idx]).sum())
                nread += len(read_idx)
                for fam, part in (("beak", "beak"), ("belly", "belly")):
                    idx = [j for j in read_idx
                           if feat_names[j].split(":")[0] == part]
                    if fam == "beak":
                        flip_beak += int((v[idx] != b["v"][idx]).sum()); nbeak += len(idx)
                    else:
                        flip_belly += int((v[idx] != b["v"][idx]).sum()); nbelly += len(idx)
                pred = int(dt.predict(v.reshape(1, -1))[0])
                nexp = backend.explain_sufficient(v, pred, sample_id=sid)
                lits = tuple(sorted((int(c.feature_idx), 1 if c.operator != "<=" else 0)
                                    for c in nexp.conditions))
                gtb = {f"{a.part}:{a.attr_name}={a.value}": int(attrs[j])
                       for j, a in enumerate(vocab)}
                pred_same += int(pred == b["pred"])
                reason_same += int(lits == b["lits"])
                if b["k1"]:
                    n_k1 += 1
                    k1_same += int(lits == b["lits"])
                else:
                    n_kgt += 1
                    kgt_same += int(lits == b["lits"])
                allc += int(all(gtb.get(feat_names[f]) == w for f, w in lits))
                agree += int(lr_model.predict(v.reshape(1, -1))[0] == pred)
                if v[yj] == 0:  # certified pattern on jittered bits
                    cert_mask_n += 1
                    cert_mask_k += int(lr_model.predict(v.reshape(1, -1))[0] == 3)
            reps.append({
                "rep": rep,
                "bit_flip_rate": flip / nread,
                "bit_flip_beak": flip_beak / max(1, nbeak),
                "bit_flip_belly": flip_belly / max(1, nbelly),
                "pred_stable": pred_same / n,
                "reason_stable": reason_same / n,
                "k1_reason_stable": k1_same / max(1, n_k1),
                "kgt_reason_stable": kgt_same / max(1, n_kgt),
                "allc_rate": allc / n,
                "overall_agreement_jittered": agree / n,
                "certified_pattern_fidelity_jittered":
                    cert_mask_k / cert_mask_n if cert_mask_n else None,
                "certified_pattern_n": cert_mask_n,
            })
        agg = {k: (float(np.mean([r[k] for r in reps])),
                   float(np.min([r[k] for r in reps])),
                   float(np.max([r[k] for r in reps])))
               for k in reps[0] if k != "rep"}
        rows_out.append({"sigma_px": sigma, "n_rep": len(reps),
                         "mean": {k: v[0] for k, v in agg.items()},
                         "min": {k: v[1] for k, v in agg.items()},
                         "max": {k: v[2] for k, v in agg.items()},
                         "reps": reps})
        m = rows_out[-1]["mean"]
        print(f"--> sigma {sigma:4.0f}px: flip {m['bit_flip_rate']:.3f} "
              f"(beak {m['bit_flip_beak']:.3f} belly {m['bit_flip_belly']:.3f}) | "
              f"pred {m['pred_stable']:.3f} reason {m['reason_stable']:.3f} "
              f"(k1 {m['k1_reason_stable']:.3f}) | allc {m['allc_rate']:.3f} | "
              f"agree {m['overall_agreement_jittered']:.3f} | "
              f"certfid {m['certified_pattern_fidelity_jittered']}", flush=True)

    report = {
        "protocol": {
            "question": ("How does the certified guarantee degrade with "
                         "box-localization noise (image unchanged, centers "
                         "jittered; pooled from cached maps)?"),
            "predictions_pre_registered": [
                "PN1 flip < 8% @4px, 10-30% @16px",
                "PN2 k=1 reason stability >= 0.9 @4px,8px",
                "PN3 k>1 stability < k=1 at every sigma",
                "PN4 certified-pattern fidelity >= 0.9 through 8px",
                "PN5 allc within 5 points of 25.4% through 8px",
                "PN6 beak flips > belly (smaller box)",
            ],
            "detector_arm": ("pending real cub_bound_full checkpoint; the only "
                             "local checkpoint (cache/cub_bound) is the mock "
                             "smoke run; same code path accepts --boxes-source "
                             "detector once available"),
            "substrate": "K=4 committed tree + seed-42 readers + GT boxes; "
                         "logreg linear reference trained once on base bits",
        },
        "baseline": {"allc": base_allc, "n": n},
        "curves": rows_out,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
