"""Falsification check for the train-cache horizontal-flip hypothesis.

Hypothesis: the TRAIN feature cache was built with the train transform
(RandomHorizontalFlip), so ~50% of cached maps are mirrored while the GT
part boxes used at pooling time are not — pooling left_wing boxes over a
mirrored map actually captures right-wing content.

Check: pool the left_wing GT box (normal) and its mirrored twin (flipped)
over each cached TRAIN map, and compare cosine similarity to a reference
left_wing vector built from the TEST cache (eval transform, no flip).
If the hypothesis holds, (cos_normal - cos_flipped) is bimodal around 0
with ~50% of samples negative (flipped box matches better). The TEST
cache under the same procedure is the control: it should be almost
entirely positive.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch
import yaml

from grounded_vision.pipeline import build_dataset
from grounded_vision.perception.regions import Region, pool_region_features


def pooled(fmap: torch.Tensor, bbox) -> np.ndarray:
    region = Region(id="left_wing", part_name="left_wing", bbox=bbox)
    out = pool_region_features(torch.from_numpy(fmap).unsqueeze(0), [region])
    v = out["left_wing"].numpy().astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def mirror(bbox):
    ymin, xmin, ymax, xmax = bbox
    return (ymin, 1.0 - xmax, ymax, 1.0 - xmin)


def box_for(ds, idx, part="left_wing"):
    sample = ds[idx]
    boxes = getattr(sample, "part_boxes", None) or ds.meta(idx)[1]
    if part not in boxes:
        return None
    sid = int(sample.sample_id)
    vis = getattr(sample, "part_visibilities", {}) or {}
    if float(vis.get(part, 1.0)) <= 0.1:
        return None
    return sid, boxes[part]


def run_split(cache_dir, split, ds, n, ref=None):
    d = np.load(os.path.join(cache_dir, f"{split}_features.npz"), allow_pickle=True)
    fmaps, sids = d["feature_maps"], d["sample_ids"].astype(int)
    sid_to_row = {int(s): i for i, s in enumerate(sids)}
    diffs, used, skipped = [], 0, 0
    for idx in range(len(ds.samples)):
        if used >= n:
            break
        got = box_for(ds, idx)
        if got is None:
            skipped += 1
            continue
        sid, bbox = got
        if sid not in sid_to_row:
            skipped += 1
            continue
        fmap = fmaps[sid_to_row[sid]]
        v_norm = pooled(fmap, bbox)
        v_flip = pooled(fmap, mirror(bbox))
        diffs.append(float(v_norm @ ref - v_flip @ ref))
        used += 1
    diffs = np.array(diffs)
    return {
        "split": split,
        "n": used,
        "skipped_no_box_or_cache": skipped,
        "frac_flipped_box_matches_better": float((diffs < 0).mean()),
        "mean_cos_diff": float(diffs.mean()),
        "frac_diff_gt_0.01": float((diffs > 0.01).mean()),
        "frac_diff_lt_-0.01": float((diffs < -0.01).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cub_bound_full.yaml")
    ap.add_argument("--n", type=int, default=1000, help="samples per split")
    ap.add_argument("--n-ref", type=int, default=500, help="test samples building the reference")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cache_dir = cfg["data"]["cache_dir"]

    ds_train = build_dataset(cfg, "train")
    ds_test = build_dataset(cfg, "test")

    # Reference left_wing vector from the TEST cache (eval transform: no flip)
    d_test = np.load(os.path.join(cache_dir, "test_features.npz"), allow_pickle=True)
    test_sids = d_test["sample_ids"].astype(int)
    test_sid_to_row = {int(s): i for i, s in enumerate(test_sids)}
    ref_vecs, used = [], 0
    for idx in range(len(ds_test.samples)):
        if used >= args.n_ref:
            break
        got = box_for(ds_test, idx)
        if got is None:
            continue
        sid, bbox = got
        if sid not in test_sid_to_row:
            continue
        ref_vecs.append(pooled(d_test["feature_maps"][test_sid_to_row[sid]], bbox))
        used += 1
    ref = np.mean(ref_vecs, axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-9)
    print(f"reference: mean left_wing vector over {used} test samples")

    report = {
        "config": args.config,
        "part": "left_wing",
        "n_ref": used,
        "train": run_split(cache_dir, "train", ds_train, args.n, ref),
        "test_control": run_split(cache_dir, "test", ds_test, args.n, ref),
    }
    out = args.out or os.path.join(cache_dir, "train_flip_check.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    for k in ("train", "test_control"):
        r = report[k]
        print(f"{r['split']}: n={r['n']}  flipped-box-better={r['frac_flipped_box_matches_better']:.1%}  "
              f"mean cos diff={r['mean_cos_diff']:.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
