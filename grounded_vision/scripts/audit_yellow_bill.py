"""
Script: audit_yellow_bill.py
============================
Pixel-level audit of the beak:bill_color=yellow dispute on the K=4
subset (Acadian Flycatcher, American Crow, American Goldfinch,
American Pipit): reader fires yellow-positive on 88/118 test images
while CUB annotates yellow absent on all 118. Who is wrong?

Measures, per species x reader-assertion:
- raw CUB annotation: present-rates of every bill_color attribute and
  the certainty distribution of has_bill_color::yellow;
- pixels in the GT beak box: fraction of yellowish pixels (HSV hue
  35-75 deg, S>=0.25, V>=0.2) and mean S/V of those pixels.

Readers use the exact scaling-run protocol (seed 42, 40 epochs) so the
yellow bit matches the committed experiments.
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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    dtr = np.load(Path(args.cache_dir) / "train_features.npz")
    dte = np.load(Path(args.cache_dir) / "test_features.npz")
    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=args.seed)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=args.seed)
    vocab = ds_train.bound_attributes
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    yj = [j for j, a in enumerate(vocab)
          if a.part == "beak" and a.attr_name == "bill_color"
          and a.value == "yellow"][0]

    img_root = Path(args.data_dir)
    class_by_dir = {}
    for line in open(img_root / "classes.txt"):
        p = line.split()
        class_by_dir[p[1]] = int(p[0])
    path_by_imgid = {}
    for line in open(img_root / "images.txt"):
        p = line.split()
        path_by_imgid[int(p[0])] = img_root / "images" / p[1]

    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])

    def pooled(rows, fmaps):
        out = {}
        fm = torch.from_numpy(fmaps.astype(np.float32))
        for i, row in enumerate(rows):
            sid, _meta_sid, boxes, vis, _attrs = row
            regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                    for p in ds_train.part_names if vis.get(p, 0.0) > 0.5 and p in boxes]
            if regs:
                p = pool_region_features(fm[i:i + 1], regs)
                out[int(sid)] = {r.part_name: p[r.id].numpy() for r in regs}
        return out

    print("--> pooling train (for readers)...", flush=True)
    tr_pooled = pooled(tr_rows, dtr["feature_maps"])
    tr_gt = {int(m[0]): m[4] for m in tr_rows}
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
    print("--> readers trained; evaluating K=4 test images", flush=True)

    def hsv_yellow_frac(img_path, bbox):
        ymin, xmin, ymax, xmax = bbox
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        box = im.crop((int(xmin * W), int(ymin * H),
                       int(np.ceil(xmax * W)), int(np.ceil(ymax * H))))
        a = np.asarray(box).astype(np.float32) / 255.0
        mx = a.max(-1); mn = a.min(-1)
        v = mx
        s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        d = np.maximum(mx - mn, 1e-6)
        h = np.zeros_like(mx)
        m = (mx == r); h[m] = (60 * ((g - b) / d) % 360)[m]
        m = (mx == g); h[m] = (60 * ((b - r) / d) + 120)[m]
        m = (mx == b); h[m] = (60 * ((r - g) / d) + 240)[m]
        ymask = (h >= 35) & (h <= 75) & (s >= 0.25) & (v >= 0.2)
        return (float(ymask.mean()),
                float(s[ymask].mean()) if ymask.any() else None,
                float(v[ymask].mean()) if ymask.any() else None,
                float(v.mean()))

    # species names for labels 0..3 in the dataset's label space
    label_names = sorted(set(
        r for r in (ds_train.class_label_names
                    if hasattr(ds_train, "class_label_names") else [])))
    # (fall back: derive from te_rows via ds class lookup if unavailable)

    rows_out = []
    for k, row in enumerate(te_rows):
        sid, meta_sid, boxes, vis, attrs = row
        if sid_to_label_te[sid] >= 4:
            continue
        pooled_te = pooled([row], dte["feature_maps"][k:k + 1])
        bit, prob = None, None
        if sid in pooled_te and "beak" in pooled_te[sid]:
            with torch.no_grad():
                t = torch.from_numpy(pooled_te[sid]["beak"].reshape(1, -1)).float()
                prob = float(torch.sigmoid(reader.classifiers[f"attr_{yj}"](t))[0, 0])
            bit = int(prob >= 0.5)
        if "beak" not in boxes or vis.get("beak", 0.0) <= 0.5:
            continue
        yf, ys, yv, vm = hsv_yellow_frac(path_by_imgid[int(meta_sid)], boxes["beak"])
        rows_out.append({
            "sid": sid, "label": sid_to_label_te[sid],
            "reader_yellow_bit": bit, "reader_yellow_prob": prob,
            "gt_yellow": int(attrs[yj] > 0),
            "yellow_px_frac": yf, "yellow_px_sat": ys,
            "yellow_px_val": yv, "mean_v": vm,
            "beak_area_px": None,
        })

    out = {"protocol": {
        "question": "Is the reader's bill_color=yellow firing reader error or "
                    "annotation vocabulary? Per-species reader firings, GT, and "
                    "yellow-pixel fractions in the GT beak box on the K=4 subset",
        "species_note": "K=4 labels 0-3 per CUBDataset ordering; names resolved "
                        "from per-species grouping below"},
        "rows": rows_out,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    # console summary
    by = defaultdict(lambda: defaultdict(list))
    for r in rows_out:
        key = "fire" if r["reader_yellow_bit"] == 1 else "nofire"
        by[r["label"]][key].append(r)
        by[r["label"]]["gt_yellow"].append(r["gt_yellow"])
    print("\nspecies(label) | n | reader-fires | GT-yellow | yellow-px frac (fire / nofire) | mean S of yellow px (fire)")
    for lab in sorted(by):
        f = by[lab]["fire"]; nf = by[lab]["nofire"]
        print(f"  label {lab}: n={len(f)+len(nf):3d} fires={len(f):3d} "
              f"GT_yellow={sum(by[lab]['gt_yellow'])} "
              f"px {np.mean([r['yellow_px_frac'] for r in f]) if f else float('nan'):.3f}"
              f"/{np.mean([r['yellow_px_frac'] for r in nf]) if nf else float('nan'):.3f} "
              f"S {np.mean([r['yellow_px_sat'] for r in f if r['yellow_px_sat'] is not None]) if any(r['yellow_px_sat'] is not None for r in f) else float('nan'):.2f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
