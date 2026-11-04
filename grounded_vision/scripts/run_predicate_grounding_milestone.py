"""
Script: run_predicate_grounding_milestone.py
============================================
First milestone of the review validation roadmap: a small set of recurring
part-attribute predicates on real CUB-200-2011, evaluated on the DISJOINT
official test split, with correct-region and matched control-region
perturbations (review requirement 2).

For every in-scope canonical predicate (e.g. breast:breast_color=yellow):

1. Agreement — logistic probe on pooled DINOv2 patch features of the
   predicate's own part box, trained on the train split, scored on the test
   split against annotated attributes (only where the part is VISIBLE:
   unobserved parts are excluded, per the three-valued semantics).
2. Targeted perturbation — blur or gray-mask the predicate's own part box,
   re-extract features, recompute the probe probability.
3. Control perturbation — the SAME edit geometry (the part's box size placed
   at another visible part's center) applied elsewhere in the image.
4. Specificity — mean |dP| under target vs control edits, plus stability of
   the same predicate under target edits of UNRELATED parts.

Pass pattern required by the review: altering the relevant region changes
the relevant predicate more than a matched control perturbation, while
unrelated predicates remain comparatively stable.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset, get_cub_transforms
from grounded_vision.perception.backbone import FrozenDINOBackbone

DEFAULT_PARTS = ["breast", "belly", "crown", "beak"]


# --------------------------------------------------------------- features ----
def pooled_patch_feature(patch_tokens: np.ndarray, box, grid: int = 16) -> np.ndarray:
    """Mean (L2-normalized) patch token over patches whose centers fall inside
    the normalized box (ymin, xmin, ymax, xmax). Falls back to the single
    patch nearest the box center if the box covers no patch center."""
    ymin, xmin, ymax, xmax = box
    rows, cols = [], []
    for r in range(grid):
        for c in range(grid):
            cx, cy = (c + 0.5) / grid, (r + 0.5) / grid
            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                rows.append(r)
                cols.append(c)
    if not rows:
        cr = int(np.clip((ymin + ymax) / 2 * grid - 0.5, 0, grid - 1))
        cc = int(np.clip((xmin + xmax) / 2 * grid - 0.5, 0, grid - 1))
        rows, cols = [cr], [cc]
    fmap = patch_tokens.reshape(grid, grid, -1)
    feat = fmap[np.ix_(rows, cols)].reshape(-1, fmap.shape[-1]).mean(axis=0)
    return feat / (np.linalg.norm(feat) + 1e-9)


@torch.no_grad()
def batch_pooled(backbone, transform, items, parts, batch_size):
    """items: list of (load_fn, boxes) where load_fn() -> PIL image (lazily
    materialized per batch to keep memory flat). Returns a list of
    {part: feature} aligned with items."""
    out = [None] * len(items)
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        images = [load().convert("RGB") for load, _ in chunk]
        batch = torch.stack([transform(im) for im in images])
        patch_tokens, _, _ = backbone.forward(batch)
        tokens = patch_tokens.cpu().numpy()
        for k, (_, boxes) in enumerate(chunk):
            feats = {}
            for p in parts:
                box = boxes.get(p)
                if box is not None:
                    feats[p] = pooled_patch_feature(tokens[k], box)
            out[start + k] = feats
        del images
    return out


# ------------------------------------------------------------------- edits ----
def denorm_box(box, w, h):
    ymin, xmin, ymax, xmax = box
    return (xmin * w, ymin * h, xmax * w, ymax * h)  # PIL xy


def apply_edit(loader, box, edit: str):
    im = loader().convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = denorm_box(box, w, h)
    if edit == "mask":
        ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    elif edit == "blur":
        side = max(4, int(0.10 * min(w, h)))
        crop = im.crop((x0, y0, x1, y1)).filter(
            ImageFilter.GaussianBlur(radius=side // 2)
        )
        im.paste(crop, (int(x0), int(y0)))
    else:
        raise ValueError(edit)
    return im


def control_box(target_box, control_center):
    """Same width/height as the target box, centered at the control part's
    box center, clipped to the image."""
    th, tw = target_box[2] - target_box[0], target_box[3] - target_box[1]
    cy, cx = control_center
    ymin = float(np.clip(cy - th / 2, 0.0, max(0.0, 1.0 - th)))
    xmin = float(np.clip(cx - tw / 2, 0.0, max(0.0, 1.0 - tw)))
    return (ymin, xmin, ymin + th, xmin + tw)


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default=",".join(DEFAULT_PARTS))
    ap.add_argument("--train-limit", type=int, default=1500)
    ap.add_argument("--test-limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pos-train-min", type=int, default=30)
    ap.add_argument("--pos-test-min", type=int, default=10)
    ap.add_argument("--acc-min", type=float, default=0.70)
    ap.add_argument("--specificity-min", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    rng = np.random.RandomState(args.seed)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=args.seed)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=args.seed)
    # eval transform for both splits: no flip — pixels and boxes must align
    ds_train.transform = get_cub_transforms(ds_train.image_size, is_train=False)
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
    print(f"--> train {len(train_idx)} / test {len(test_idx)} images; "
          f"scope parts: {parts}", flush=True)

    vocab = ds_train.bound_attributes  # BoundAttribute(attr_id, part, attr, value)
    predicates = [j for j, a in enumerate(vocab) if a.part in parts]
    print(f"--> {len(predicates)} in-scope predicates: "
          + ", ".join(f"{vocab[j].part}:{vocab[j].attr_name}={vocab[j].value}"
                      for j in predicates), flush=True)

    backbone = FrozenDINOBackbone()
    print(f"--> backbone device: {backbone.device} (mock={backbone.is_mock})",
          flush=True)

    def loader_for(ds, i):
        path = ds.samples[int(i)].image_path
        return lambda: Image.open(path)

    # ---- original features ---------------------------------------------------
    def orig_items(ds, metas, indices):
        items, sids = [], []
        for k, i in enumerate(indices):
            sid, vboxes, _ = metas[k]
            if not vboxes:
                continue
            items.append((loader_for(ds, i), vboxes))
            sids.append(int(sid))
        return items, sids

    train_items, train_sids = orig_items(ds_train, train_meta, train_idx)
    test_items, test_sids = orig_items(ds_test, test_meta, test_idx)
    train_feats = dict(zip(train_sids, batch_pooled(
        backbone, transform, train_items, parts, args.batch_size)))
    test_feats = dict(zip(test_sids, batch_pooled(
        backbone, transform, test_items, parts, args.batch_size)))
    print(f"--> original features: {len(train_feats)} train / "
          f"{len(test_feats)} test images", flush=True)

    train_labels = {int(m[0]): m[2] for m in train_meta}
    test_labels = {int(m[0]): m[2] for m in test_meta}

    # ---- probes + recurring-predicate filter ---------------------------------
    probes, kept = {}, []
    for j in predicates:
        part = vocab[j].part
        X, y = [], []
        for sid, feats in train_feats.items():
            if part in feats:
                X.append(feats[part])
                y.append(int(train_labels[sid][j]))
        y = np.array(y)
        n_pos_tr = int(y.sum())
        n_pos_te = sum(int(test_labels[sid][j])
                       for sid, feats in test_feats.items() if part in feats)
        if n_pos_tr < args.pos_train_min or n_pos_te < args.pos_test_min:
            continue
        probe = LogisticRegression(max_iter=2000, class_weight="balanced")
        probe.fit(np.array(X), y)
        probes[j] = probe
        kept.append((j, n_pos_tr, int(len(y) - n_pos_tr), n_pos_te))
    print(f"--> recurring predicates kept ({len(kept)}): "
          + ", ".join(f"{vocab[j].part}:{vocab[j].attr_name}={vocab[j].value}"
                      f"[tr {p}+/{n}-, te {pt}+]" for j, p, n, pt in kept),
          flush=True)

    def p_of(j, feat):
        return float(probes[j].predict_proba(feat.reshape(1, -1))[0, 1])

    P0 = {j: {} for j, _, _, _ in kept}
    for sid, feats in test_feats.items():
        for j, _, _, _ in kept:
            part = vocab[j].part
            if part in feats:
                P0[j][sid] = p_of(j, feats[part])

    # ---- perturbation variants (single combined pass) ------------------------
    edits = ["blur", "mask"]
    target_items, target_keys = [], []   # keys: (sid, edit, part_edited)
    control_items, control_keys = [], []
    n_two_part = 0
    for k, i in enumerate(test_idx):
        sid, vboxes, _ = test_meta[k]
        if len(vboxes) < 2:  # need a control part center
            continue
        n_two_part += 1
        load = loader_for(ds_test, i)
        order = [p for p in parts if p in vboxes]
        for p in order:
            ctrl = order[(order.index(p) + 1) % len(order)]
            for e in edits:
                target_items.append(
                    (lambda l=load, b=vboxes[p], ed=e: apply_edit(l, b, ed),
                     vboxes))
                target_keys.append((sid, e, p))
                cbox = control_box(vboxes[p], box_center(vboxes[ctrl]))
                control_items.append(
                    (lambda l=load, b=cbox, ed=e: apply_edit(l, b, ed),
                     vboxes))
                control_keys.append((sid, e, p))

    print(f"--> perturbation pass: {len(target_items)} target + "
          f"{len(control_items)} control edits over {n_two_part} images",
          flush=True)
    target_feats = dict(zip(target_keys, batch_pooled(
        backbone, transform, target_items, parts, args.batch_size)))
    control_feats = dict(zip(control_keys, batch_pooled(
        backbone, transform, control_items, parts, args.batch_size)))

    # ---- metrics --------------------------------------------------------------
    records = []
    for j, n_pos_tr, n_neg_tr, _ in kept:
        part = vocab[j].part
        ys = np.array([int(test_labels[sid][j]) for sid in P0[j]])
        ps = np.array([P0[j][sid] for sid in P0[j]])
        acc = float(np.mean((ps > 0.5).astype(int) == ys))
        entry = {
            "feature": f"{vocab[j].part}:{vocab[j].attr_name}={vocab[j].value}",
            "part": part,
            "cub_attr_id": (int(ds_train.attribute_cub_ids[j])
                            if ds_train.attribute_cub_ids else None),
            "n_train": {"pos": n_pos_tr, "neg": n_neg_tr},
            "agreement": {
                "n_test": int(len(ys)),
                "accuracy": acc,
                "auroc": (float(roc_auc_score(ys, ps))
                          if 0 < ys.sum() < len(ys) else None),
            },
        }

        pert = {}
        for e in edits:
            t_signed = [p_of(j, target_feats[(sid, e, part)][part]) - P0[j][sid]
                        for (sid, ee, p) in target_keys
                        if ee == e and p == part and sid in P0[j]
                        and part in target_feats[(sid, e, part)]]
            c_abs = [abs(p_of(j, control_feats[(sid, e, part)][part]) - P0[j][sid])
                     for (sid, ee, p) in control_keys
                     if ee == e and p == part and sid in P0[j]
                     and part in control_feats[(sid, e, part)]]
            u_abs = [abs(p_of(j, target_feats[(sid, e, p)][part]) - P0[j][sid])
                     for (sid, ee, p) in target_keys
                     if ee == e and p != part and sid in P0[j]
                     and part in target_feats[(sid, e, p)]]

            t_abs = [abs(d) for d in t_signed]
            m_t = float(np.mean(t_abs)) if t_abs else None
            m_c = float(np.mean(c_abs)) if c_abs else None
            pert[e] = {
                "mean_abs_dp_target": m_t,
                "mean_signed_dp_target": float(np.mean(t_signed)) if t_signed else None,
                "target_flip_rate": float(np.mean(
                    [(p_of(j, target_feats[(sid, e, part)][part]) > 0.5)
                     != (P0[j][sid] > 0.5)
                     for (sid, ee, p) in target_keys
                     if ee == e and p == part and sid in P0[j]])) if t_signed else None,
                "mean_abs_dp_control": m_c,
                "mean_abs_dp_unrelated": float(np.mean(u_abs)) if u_abs else None,
                "specificity_ratio": (m_t / m_c
                                      if m_t is not None and m_c not in (None, 0.0)
                                      else None),
            }
        entry["perturbation"] = pert
        entry["passes"] = bool(
            acc >= args.acc_min
            and all(pert[e]["specificity_ratio"] is not None
                    and pert[e]["specificity_ratio"] >= args.specificity_min
                    for e in edits)
        )
        records.append(entry)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_in_scope": len(predicates),
        "n_predicates": len(records),
        "mean_accuracy": _mean([r["agreement"]["accuracy"] for r in records]),
        "mean_specificity_ratio": {
            e: _mean([r["perturbation"][e]["specificity_ratio"] for r in records])
            for e in edits},
        "n_passing": int(sum(r["passes"] for r in records)),
        "decision_rule": (f"predicate passes if agreement accuracy >= "
                          f"{args.acc_min} and specificity ratio >= "
                          f"{args.specificity_min} under BOTH edits"),
    }

    report = {
        "params": {
            "data_dir": args.data_dir,
            "parts": parts,
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "pos_train_min": args.pos_train_min,
            "pos_test_min": args.pos_test_min,
            "acc_min": args.acc_min,
            "specificity_min": args.specificity_min,
            "seed": args.seed,
            "backbone": "dinov2_vits14",
            "probe": "logistic_regression_balanced_on_part_pooled_features",
            "box_policy": ("CUBDataset template boxes centered on annotated "
                           "part points; control box = target box size at the "
                           "next visible scope part's center"),
            "edits": edits,
        },
        "predicates": records,
        "summary": summary,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {args.out}", flush=True)
    print(f"--> predicates: {summary['n_predicates']} | mean acc: "
          f"{summary['mean_accuracy']:.3f} | passing: {summary['n_passing']}")
    for r in records:
        line = f"  {r['feature']:44s} acc={r['agreement']['accuracy']:.3f} "
        for e in edits:
            pe = r["perturbation"][e]
            line += (f"| {e}: T={pe['mean_abs_dp_target']}"
                     f" C={pe['mean_abs_dp_control']}"
                     f" S={pe['specificity_ratio']} ")
        print(line + ("PASS" if r["passes"] else "fail"), flush=True)


if __name__ == "__main__":
    main()
