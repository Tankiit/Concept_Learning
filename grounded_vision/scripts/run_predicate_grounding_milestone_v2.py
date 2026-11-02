"""
Script: run_predicate_grounding_milestone_v2.py
===============================================
Milestone v2 of the review validation roadmap. Extends v1
(run_predicate_grounding_milestone.py) with:

1. DEPLOYED READERS - the framework's AttributeReader heads, trained with
   the fit_perception.py protocol (per-region linear heads, BCE-with-logits
   with pos_weight = neg/pos clamped to 50, Adam lr 1e-3, 40 epochs, seed
   42) on the same train split, evaluated alongside the v1 logistic probes.
   Features for BOTH instruments come from the framework's own
   pool_region_features over GT part boxes.
2. RICHER CONTROLS - three control families per target part:
   - adjacent:   same-size box at the next visible scope part's center (the
                 v1 rule; anatomically adjacent for crown)
   - farthest:   same-size box at the FARTHEST visible scope part's center
   - dist_matched: same-size box placed at the ADJACENT control's distance
                 from the target, in the direction (of 8) that maximizes the
                 minimum distance to any visible part center - same distance,
                 different content. Isolates distance from part identity.
3. PER-IMAGE EFFECTS - every (image, predicate, edit, instrument) record
   with p0 and signed deltas under each condition is saved as JSONL, and the
   summary reports bootstrap 95% CIs (1000 resamples) on all specificity
   ratios, so uncertainty is estimated rather than asserted.

Decision question: does the v1 target-vs-control separation persist under
the deployed readers, and does region geometry (control adjacency) explain
the crown/breast failure?
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
from PIL import Image, ImageDraw, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset, get_cub_transforms
from grounded_vision.perception.backbone import FrozenDINOBackbone
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader

DEFAULT_PARTS = ["breast", "belly", "crown", "beak"]
EDITS = ["blur", "mask"]


# ------------------------------------------------------------------ edits ----
def apply_edit(loader, box, edit):
    im = loader().convert("RGB")
    w, h = im.size
    ymin, xmin, ymax, xmax = box
    x0, y0, x1, y1 = xmin * w, ymin * h, xmax * w, ymax * h
    if edit == "mask":
        ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    else:
        side = max(4, int(0.10 * min(w, h)))
        crop = im.crop((x0, y0, x1, y1)).filter(
            ImageFilter.GaussianBlur(radius=side // 2))
        im.paste(crop, (int(x0), int(y0)))
    return im


def same_size_box_at(target_box, center):
    th, tw = target_box[2] - target_box[0], target_box[3] - target_box[1]
    cy, cx = center
    ymin = float(np.clip(cy - th / 2, 0.0, max(0.0, 1.0 - th)))
    xmin = float(np.clip(cx - tw / 2, 0.0, max(0.0, 1.0 - tw)))
    return (ymin, xmin, ymin + th, xmin + tw)


def dist_matched_box(target_box, dist, centers):
    """Same-size box at `dist` from the target center, in the direction (of
    8) whose FINAL (clipped) placement maximizes min distance to any part
    center. Returns (box, achieved_dist)."""
    tcy, tcx = (target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2
    th, tw = target_box[2] - target_box[0], target_box[3] - target_box[1]
    best, best_score = None, -1.0
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        cy = float(np.clip(tcy + dist * np.sin(ang), th / 2, 1 - th / 2))
        cx = float(np.clip(tcx + dist * np.cos(ang), tw / 2, 1 - tw / 2))
        mind = min((cy - py) ** 2 + (cx - px) ** 2 for py, px in centers)
        if mind > best_score:
            best_score, best = mind, (cy, cx)
    box = same_size_box_at(target_box, best)
    achieved = float(np.hypot(best[0] - tcy, best[1] - tcx))
    return box, achieved


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def center_dist(b1, b2):
    a, b = box_center(b1), box_center(b2)
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ------------------------------------------------------------- features ------
@torch.no_grad()
def batch_part_features(backbone, transform, items, parts, batch_size):
    """items: list of (load_fn, visible_boxes_dict). Returns list aligned with
    items of {part: np.ndarray[384]} pooled via the framework's
    pool_region_features (identical to the deployed pipeline)."""
    out = [None] * len(items)
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        images = [load().convert("RGB") for load, _ in chunk]
        batch = torch.stack([transform(im) for im in images])
        tokens, _, _ = backbone.forward(batch)
        fmap = (tokens.reshape(tokens.shape[0], 16, 16, -1)
                .permute(0, 3, 1, 2).cpu())  # [B, D, 16, 16]
        for k, (_, vboxes) in enumerate(chunk):
            regions = [Region(id=f"region_{p}", part_name=p, bbox=b)
                       for p, b in vboxes.items()]
            pooled = pool_region_features(fmap[k : k + 1], regions)
            out[start + k] = {p: pooled[f"region_{p}"].numpy()
                              for p in vboxes}
        del images
    return out


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parts", default=",".join(DEFAULT_PARTS))
    ap.add_argument("--train-limit", type=int, default=1500)
    ap.add_argument("--test-limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pos-train-min", type=int, default=30)
    ap.add_argument("--pos-test-min", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--acc-min", type=float, default=0.70)
    ap.add_argument("--specificity-min", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=args.seed)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=args.seed)
    ds_train.transform = get_cub_transforms(224, is_train=False)  # no flip
    transform = ds_test.transform

    def subsample(ds, limit):
        idx = np.arange(len(ds.samples))
        if limit and limit < len(idx):
            idx = np.sort(rng.permutation(idx)[:limit])
        return idx

    train_idx = subsample(ds_train, args.train_limit)
    test_idx = subsample(ds_test, args.test_limit)

    def meta(ds, i):
        sid, boxes, vis, attrs = ds.meta(int(i))
        vboxes = {p: boxes[p] for p in parts
                  if vis.get(p, 0.0) > 0.5 and p in boxes}
        return sid, vboxes, attrs

    train_meta = [meta(ds_train, i) for i in train_idx]
    test_meta = [meta(ds_test, i) for i in test_idx]
    print(f"--> train {len(train_idx)} / test {len(test_idx)}; parts {parts}",
          flush=True)

    vocab = ds_train.bound_attributes  # BoundAttribute(attr_id, part, attr, value)
    predicates = [j for j, a in enumerate(vocab) if a.part in parts]
    assert all(vocab[j].attr_id == j for j in range(len(vocab))), \
        "vocabulary local ids must equal list positions (reader head keys)"

    def loader_for(ds, i):
        path = ds.samples[int(i)].image_path
        return lambda: Image.open(path)

    backbone = FrozenDINOBackbone()
    print(f"--> backbone {backbone.device} (mock={backbone.is_mock})",
          flush=True)

    def orig_items(metas, ds, indices):
        items, sids = [], []
        for k, i in enumerate(indices):
            if metas[k][1]:
                items.append((loader_for(ds, i), metas[k][1]))
                sids.append(int(metas[k][0]))
        return items, sids

    tr_items, tr_sids = orig_items(train_meta, ds_train, train_idx)
    te_items, te_sids = orig_items(test_meta, ds_test, test_idx)
    train_feats = dict(zip(tr_sids, batch_part_features(
        backbone, transform, tr_items, parts, args.batch_size)))
    test_feats = dict(zip(te_sids, batch_part_features(
        backbone, transform, te_items, parts, args.batch_size)))
    print(f"--> features: {len(train_feats)} train / {len(test_feats)} test",
          flush=True)

    train_labels = {int(m[0]): m[2] for m in train_meta}
    test_labels = {int(m[0]): m[2] for m in test_meta}

    # ---- recurring-predicate filter ------------------------------------------
    kept = []
    for j in predicates:
        part = vocab[j].part
        pos_tr = sum(int(train_labels[s][j]) for s, f in train_feats.items()
                     if part in f)
        pos_te = sum(int(test_labels[s][j]) for s, f in test_feats.items()
                     if part in f)
        n_tr = sum(1 for s, f in train_feats.items() if part in f)
        if pos_tr >= args.pos_train_min and pos_te >= args.pos_test_min:
            kept.append((j, pos_tr, n_tr - pos_tr, pos_te))
    print(f"--> recurring predicates ({len(kept)}): "
          + ", ".join(f"{vocab[j].part}:{vocab[j].attr_name}={vocab[j].value}"
                      for j, *_ in kept), flush=True)

    # ---- instrument 1: logistic probes (v1 protocol, framework pooling) ------
    probes = {}
    for j, *_ in kept:
        part = vocab[j].part
        X = np.array([f[part] for s, f in sorted(train_feats.items())
                      if part in f])
        y = np.array([int(train_labels[s][j])
                      for s, f in sorted(train_feats.items()) if part in f])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(X, y)
        probes[j] = lr

    # ---- instrument 2: deployed AttributeReader (fit_perception protocol) ----
    attrs_by_part = {}
    for j, a in enumerate(vocab):
        attrs_by_part.setdefault(a.part, []).append(j)
    region_rows = {}
    for s, feats in train_feats.items():
        for part, vec in feats.items():
            region_rows.setdefault(part, []).append((s, vec))
    reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
    region_data = {}
    for part, rows in region_rows.items():
        cols = attrs_by_part.get(part)
        if not cols:
            continue
        X = torch.from_numpy(np.stack([v for _, v in rows])).float()
        y = torch.from_numpy(np.array(
            [train_labels[s][cols] for s, _ in rows], dtype=np.float32))
        pos, neg = y.sum(dim=0), y.shape[0] - y.sum(dim=0)
        pw = (neg / pos.clamp(min=1.0)).clamp(max=50.0)
        region_data[part] = (X, y, pw, cols)
    for epoch in range(args.epochs):
        loss = 0.0
        for part, (X, y, pw, cols) in region_data.items():
            W = torch.cat([reader.classifiers[f"attr_{j}"].weight
                           for j in cols], dim=0)
            b = torch.cat([reader.classifiers[f"attr_{j}"].bias
                           for j in cols], dim=0)
            logits = X @ W.T + b
            loss = loss + F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"   reader epoch {epoch + 1}/{args.epochs} "
                  f"loss {float(loss):.4f}", flush=True)

    def p_of(instrument, j, feat):
        if instrument == "probe":
            return float(probes[j].predict_proba(
                feat.reshape(1, -1))[0, 1])
        head = reader.classifiers[f"attr_{j}"]
        t = torch.from_numpy(feat.reshape(1, -1)).float()
        return float(torch.sigmoid(head(t))[0, 0].item())

    # ---- perturbation variants ------------------------------------------------
    # key: (sid, edit, part) -> item; conditions computed per (sid, part)
    variant_items, variant_meta = [], []
    ctrl_distances = {"adjacent": [], "farthest": [], "dist_matched": []}
    for k, i in enumerate(test_idx):
        sid, vboxes, _ = test_meta[k]
        if len(vboxes) < 2:
            continue
        load = loader_for(ds_test, i)
        order = [p for p in parts if p in vboxes]
        centers_all = [box_center(vboxes[p]) for p in order]
        for p in order:
            adjacent_q = order[(order.index(p) + 1) % len(order)]
            dists = {q: center_dist(vboxes[p], vboxes[q])
                     for q in order if q != p}
            far_q = max(dists, key=dists.get)
            adj_box = same_size_box_at(vboxes[p], box_center(vboxes[adjacent_q]))
            far_box = same_size_box_at(vboxes[p], box_center(vboxes[far_q]))
            dm_box, dm_ach = dist_matched_box(
                vboxes[p], center_dist(vboxes[p], vboxes[adjacent_q]),
                centers_all)
            ctrl_distances["adjacent"].append(
                center_dist(vboxes[p], adj_box))
            ctrl_distances["farthest"].append(
                center_dist(vboxes[p], far_box))
            ctrl_distances["dist_matched"].append(dm_ach)
            plan = [("target", vboxes[p]), ("adjacent", adj_box),
                    ("farthest", far_box), ("dist_matched", dm_box)]
            for cond, box in plan:
                for e in EDITS:
                    variant_items.append(
                        (lambda l=load, b=box, ed=e: apply_edit(l, b, ed),
                         vboxes))
                    variant_meta.append((sid, e, p, cond))
    print(f"--> {len(variant_items)} perturbation variants over "
          f"{len({m[0] for m in variant_meta})} images", flush=True)
    variant_feats = batch_part_features(backbone, transform, variant_items,
                                        parts, args.batch_size)
    V = {}
    for (sid, e, p, cond), feats in zip(variant_meta, variant_feats):
        V[(sid, e, p, cond)] = feats
    print("--> control placement mean distances: "
          + ", ".join(f"{c}={np.mean(v):.3f}"
                      for c, v in ctrl_distances.items()), flush=True)

    # ---- per-image effects ------------------------------------------------------
    instruments = ["probe", "reader"]
    effects_path = out_dir / "predicate_grounding_milestone_v2_effects.jsonl"
    records = {inst: {j: {e: [] for e in EDITS} for j, *_ in kept}
               for inst in instruments}
    with effects_path.open("w") as ef:
        for j, *_ in kept:
            part = vocab[j].part
            for sid, feats in test_feats.items():
                if part not in feats:
                    continue
                for inst in instruments:
                    p0 = p_of(inst, j, feats[part])
                    for e in EDITS:
                        tgt = V.get((sid, e, part, "target"), {}).get(part)
                        adj = V.get((sid, e, part, "adjacent"), {}).get(part)
                        far = V.get((sid, e, part, "farthest"), {}).get(part)
                        dmm = V.get((sid, e, part, "dist_matched"), {}).get(part)
                        if tgt is None or adj is None:
                            continue
                        unrel = [p_of(inst, j, V[(sid, e, q, "target")][part])
                                 - p0
                                 for q in parts
                                 if q != part and (sid, e, q, "target") in V
                                 and part in V[(sid, e, q, "target")]]
                        rec = {
                            "sample_id": sid,
                            "feature": f"{vocab[j].part}:{vocab[j].attr_name}={vocab[j].value}",
                            "part": part, "edit": e, "instrument": inst,
                            "y": int(test_labels[sid][j]), "p0": p0,
                            "dp_target": p_of(inst, j, tgt) - p0,
                            "dp_ctrl_adjacent": p_of(inst, j, adj) - p0,
                            "dp_ctrl_farthest": (p_of(inst, j, far) - p0)
                            if far is not None else None,
                            "dp_ctrl_dist_matched": (p_of(inst, j, dmm) - p0)
                            if dmm is not None else None,
                            "dp_unrelated_mean": float(np.mean(unrel))
                            if unrel else None,
                        }
                        ef.write(json.dumps(rec) + "\n")
                        records[inst][j][e].append(rec)

    # ---- summary with bootstrap CIs --------------------------------------------
    def boot_ratio(t, c, n=args.n_bootstrap, seed=1234):
        t, c = np.asarray(t, float), np.asarray(c, float)
        if len(t) == 0 or len(c) == 0:
            return None
        r = np.mean(t) / np.mean(c) if np.mean(c) > 0 else None
        if r is None:
            return None
        brng = np.random.RandomState(seed)
        n_ = len(t)
        vals = []
        for _ in range(n):
            idx = brng.randint(0, n_, n_)
            mt, mc = np.mean(np.abs(t)[idx]), np.mean(np.abs(c)[idx])
            if mc > 0:
                vals.append(mt / mc)
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (None, None))
        return {"ratio": float(r),
                "ci95": [float(lo), float(hi)] if lo is not None else None}

    summary_rows = []
    for inst in instruments:
        for j, n_pos_tr, n_neg_tr, _ in kept:
            part = vocab[j].part
            row = {
                "instrument": inst,
                "feature": f"{vocab[j].part}:{vocab[j].attr_name}={vocab[j].value}",
                "part": part, "n_train": {"pos": n_pos_tr, "neg": n_neg_tr},
            }
            recs = records[inst][j][EDITS[0]]  # agreement from any edit set
            ys = np.array([r["y"] for r in recs])
            ps = np.array([r["p0"] for r in recs])
            row["agreement"] = {
                "n_test": int(len(ys)),
                "accuracy": float(np.mean((ps > 0.5).astype(int) == ys)),
                "auroc": (float(roc_auc_score(ys, ps))
                          if 0 < ys.sum() < len(ys) else None),
            }
            pert = {}
            for e in EDITS:
                rs = records[inst][j][e]
                t = [abs(r["dp_target"]) for r in rs]
                ev = {"mean_abs_dp_target": float(np.mean(t))}
                for cond, key in (("adjacent", "dp_ctrl_adjacent"),
                                  ("farthest", "dp_ctrl_farthest"),
                                  ("dist_matched", "dp_ctrl_dist_matched")):
                    c = [abs(r[key]) for r in rs if r[key] is not None]
                    ev[f"mean_abs_dp_ctrl_{cond}"] = (
                        float(np.mean(c)) if c else None)
                    ev[f"specificity_{cond}"] = boot_ratio(t, c)
                u = [abs(r["dp_unrelated_mean"]) for r in rs
                     if r["dp_unrelated_mean"] is not None]
                ev["mean_abs_dp_unrelated"] = float(np.mean(u)) if u else None
                pert[e] = ev
            row["perturbation"] = pert
            row["passes_by_control"] = {
                cond: bool(
                    row["agreement"]["accuracy"] >= args.acc_min
                    and all(
                        (pert[e][f"specificity_{cond}"] or {}).get("ratio")
                        is not None
                        and pert[e][f"specificity_{cond}"]["ratio"]
                        >= args.specificity_min
                        for e in EDITS))
                for cond in ("adjacent", "farthest", "dist_matched")}
            summary_rows.append(row)

    by_family = {}
    for r in summary_rows:
        fam = (r["instrument"], r["part"])
        by_family.setdefault(fam, []).append(r)
    family_summary = []
    for (inst, part), rows_ in sorted(by_family.items()):
        entry = {"instrument": inst, "part": part, "n_predicates": len(rows_)}
        for cond in ("adjacent", "farthest", "dist_matched"):
            ratios = [r["perturbation"][e][f"specificity_{cond}"]["ratio"]
                      for r in rows_ for e in EDITS
                      if (r["perturbation"][e][f"specificity_{cond}"] or {})
                      .get("ratio") is not None]
            entry[f"mean_specificity_{cond}"] = (
                float(np.mean(ratios)) if ratios else None)
            entry[f"n_pass_{cond}"] = int(
                sum(r["passes_by_control"][cond] for r in rows_))
        entry["mean_accuracy"] = float(np.mean(
            [r["agreement"]["accuracy"] for r in rows_]))
        family_summary.append(entry)

    report = {
        "params": {
            "data_dir": args.data_dir, "parts": parts,
            "train_limit": args.train_limit, "test_limit": args.test_limit,
            "pos_train_min": args.pos_train_min,
            "pos_test_min": args.pos_test_min,
            "reader_training": ("AttributeReader per-region linear heads; "
                                f"BCE-with-logits pos_weight, Adam lr={args.lr}, "
                                f"{args.epochs} epochs, seed {args.seed} "
                                "(fit_perception protocol, reader-only)"),
            "pooling": "framework pool_region_features over GT part boxes",
            "controls": {
                "adjacent": "same-size box at next visible scope part center (v1 rule)",
                "farthest": "same-size box at farthest visible scope part center",
                "dist_matched": "same-size box at the adjacent-control distance, "
                                "direction maximizing min distance to part centers",
            },
            "control_mean_distances": {
                c: float(np.mean(v)) for c, v in ctrl_distances.items()},
            "bootstrap": f"{args.n_bootstrap} resamples, seed 1234",
            "edits": EDITS, "seed": args.seed,
        },
        "predicates": summary_rows,
        "family_summary": family_summary,
    }
    out = out_dir / "predicate_grounding_milestone_v2.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved: {out}")
    print(f"saved: {effects_path}")
    print("\n=== family summary (acc | specificity adj/far/dist-matched | passes) ===")
    for e in family_summary:
        print(f"  {e['instrument']:7s} {e['part']:7s} "
              f"acc={e['mean_accuracy']:.3f} "
              f"S_adj={e['mean_specificity_adjacent']:.2f} "
              f"S_far={e['mean_specificity_farthest']:.2f} "
              f"S_dm={e['mean_specificity_dist_matched']:.2f} "
              f"pass {e['n_pass_adjacent']}/{e['n_predicates']} | "
              f"{e['n_pass_farthest']}/{e['n_predicates']} | "
              f"{e['n_pass_dist_matched']}/{e['n_predicates']}", flush=True)


if __name__ == "__main__":
    main()
