#!/usr/bin/env python3
"""Beak strip (supplement): four beak crops where the bill_color=yellow
reader fires, spanning the yellow-pixel distribution (min / quartile /
median / max), each labelled with species, measured yellow-pixel share and
the reader's probability. Deterministic selection from committed data.

Sources: results/yellow_bill_audit.json (+ CUB images/boxes locally).
Output: results/figures/fig_beakstrip.{png,pdf}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from grounded_vision.data.cub import CUBDataset  # noqa: E402

DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
OUT = os.path.join(REPO, "results", "figures", "fig_beakstrip")
RED = "#c0392b"


def main():
    if os.path.exists(f"{OUT}.pdf"):
        raise SystemExit(f"REFUSING to overwrite {OUT}.pdf")
    rows = [r for r in json.load(open(os.path.join(REPO, "results",
                "yellow_bill_audit.json")))["rows"] if r["reader_yellow_bit"] == 1]
    rows.sort(key=lambda r: r["yellow_px_frac"])
    pick = [rows[0], rows[len(rows) // 4], rows[len(rows) // 2], rows[-1]]

    ds = CUBDataset(data_dir=DATA, split="test", use_mock_if_missing=False)
    id2path = dict(l.strip().split(" ", 1)
                   for l in open(os.path.join(DATA, "images.txt")))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, r in zip(axes, pick):
        k = next(i for i in range(len(ds)) if ds.meta(i)[0] == r["sid"])
        _, boxes, _, _ = ds.meta(k)
        img = plt.imread(os.path.join(DATA, "images", id2path[str(r["sid"])]))
        H, W = img.shape[:2]
        x, y, w, h = boxes["beak"]
        x0, y0 = max(0, int(x * W)), max(0, int(y * H))
        x1, y1 = min(W, int((x + w) * W)), min(H, int((y + h) * H))
        ax.imshow(img[y0:y1, x0:x1])
        ax.add_patch(Rectangle((0, 0), (x1 - x0) - 1, (y1 - y0) - 1,
                               fill=False, ec=RED, ls="--", lw=1.6))
        ax.axis("off")
        species = id2path[str(r["sid"])].split("/")[0].split(".", 1)[1].replace("_", " ")
        ax.set_title(f"{species}\n{100 * r['yellow_px_frac']:.0f}% yellow · "
                     f"p(yellow)={r['reader_yellow_prob']:.2f}\nannotation: not yellow",
                     fontsize=6.2, color=RED)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT}.pdf | picked sids {[r['sid'] for r in pick]}")


if __name__ == "__main__":
    main()
