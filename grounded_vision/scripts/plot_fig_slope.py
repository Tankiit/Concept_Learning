#!/usr/bin/env python3
"""fig:slope (SUPPLEMENT) -- vocabulary-vs-perception ladder, one line per bed.

x: head input rung (predicted bits -> pooled features -> continuous scores
-> ground-truth bits); y: accuracy of the MATCHED head family (linear/
logistic; the CART-on-GT-bits collapse is a different head family, stated
in the caption, not plotted). BDD-OIA continuous rung not run (gap in the
line). Sources: design_b_v2.json (CUB, K=4); bdd_oia_experiments.json
stage6_ladder (BDD, forward action).
Output: results/figures/fig_slope.{png,pdf}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "figures", "fig_slope")
RED, GREY = "#c0392b", "#556"
RUNGS = ["predicted\nbits", "pooled\nfeatures", "continuous\nscores", "ground-truth\nbits"]
CUB = [0.627, 0.712, 0.746, 0.805]
BDD = [0.716, 0.753, None, 0.986]   # forward action; continuous rung not run


def plot_line(ax, ys, color, label):
    xs = [i for i, y in enumerate(ys) if y is not None]
    vs = [y for y in ys if y is not None]
    ax.plot(xs, vs, "o-", color=color, ms=4, label=label)


def main():
    for ext in ("png", "pdf"):
        if os.path.exists(f"{OUT}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {OUT}.{ext}")
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    plot_line(ax, CUB, RED, "CUB (K=4)")
    plot_line(ax, BDD, GREY, "BDD-OIA (forward)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(RUNGS, fontsize=6)
    ax.set_ylabel("matched-head accuracy")
    ax.set_ylim(0.55, 1.02)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
