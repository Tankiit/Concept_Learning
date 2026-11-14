#!/usr/bin/env python3
"""fig:mechanism -- the depth ablation at K=20: support vs |R| against the
independence baseline. Points: tree depths (labelled); dashed curve: the
baseline prediction at the same |R|. Points on the curve = interventional
confirmation of per-literal compounding. Source: ablation_depth_K20.json.
Output: results/figures/fig_mechanism.{png,pdf}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "figures", "fig_mechanism")
RED, GREY = "#c0392b", "#556"


def main():
    for ext in ("png", "pdf"):
        if os.path.exists(f"{OUT}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {OUT}.{ext}")
    cells = json.load(open(os.path.join(
        REPO, "results", "ablation_depth_K20.json")))["cells"]
    x = [c["mean_len_R"] for c in cells]
    sup = [c["support"] for c in cells]
    nl = [c["null"] for c in cells]
    acc = [c["acc"] for c in cells]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(x, nl, "--", color=GREY, lw=1.3,
            label="independence baseline")
    ax.plot(x, sup, "o", color=RED, ms=6, label="measured support")
    for xi, yi, c in zip(x, sup, cells):
        ax.annotate(f"d={c['depth']}\nacc {c['acc']:.2f}", (xi, yi),
                    textcoords="offset points", xytext=(6, -12), fontsize=5.6,
                    color=RED if c["depth"] <= 6 else "#8a3030")
    ax.set_xlabel(r"mean reason length $|R|$  (tree depth knob, $K=20$ fixed)")
    ax.set_ylabel("premise support")
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
