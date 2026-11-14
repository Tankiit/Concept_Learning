#!/usr/bin/env python3
"""fig:abstain -- support vs coverage per trust channel, CUB K=20 and K=200.

Two panels (Setting T). Curves: B calibrated min-confidence (red), C raw
(grey), A_ir oracle marginal filter (dashed green), D_random = flat at the
unconditional rate. Reference lines: unconditional support R and the 2x
repair bar (pre-registered). Marker: the 0.14 operating point on B.
Sources: task1_abstention_K20_50.json, task1_abstention_K200.json.
Output: results/figures/fig_abstain.{png,pdf}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "figures", "fig_abstain")
RED, GREY, GREEN = "#c0392b", "#777", "#1a8f3c"


def curves(block):
    covs = sorted(float(c) for c in block["at_coverage"])
    out = {}
    for ch in ("B", "C", "A_ir"):
        out[ch] = [block["at_coverage"][f"{c:.3f}"][ch]["joint"] for c in covs]
    return covs, out


def panel(ax, block, title):
    covs, ch = curves(block)
    R = block["raw_joint"]
    ax.axhline(R, color=GREY, lw=0.9, ls=":")
    ax.axhline(2 * R, color="#222", lw=0.9, ls="--")
    ax.text(0.505, R, f"unconditional R = {R:.3f}", fontsize=5.8, color=GREY, va="center")
    ax.text(0.505, 2 * R, f"2x repair bar = {2 * R:.3f}", fontsize=5.8, va="center")
    ax.plot(covs, ch["A_ir"], "^--", color=GREEN, ms=4, lw=1.0, label="oracle marginal (A)")
    ax.plot(covs, ch["C"], "s-", color=GREY, ms=3.5, lw=1.1, label="raw confidence (C)")
    ax.plot(covs, ch["B"], "o-", color=RED, ms=4, lw=1.6,
            label="calibrated min-trust (B)")
    i14 = covs.index(0.14)
    ax.plot([0.14], [ch["B"][i14]], "o", ms=9, mfc="none", mec=RED, mew=1.4)
    ax.annotate("operating point\n(coverage 0.14)", (0.14, ch["B"][i14]),
                textcoords="offset points", xytext=(8, 10), fontsize=5.8, color=RED)
    ax.set_xlabel("coverage")
    ax.set_title(title, fontsize=7.5)
    ax.set_xlim(0.03, 0.56)


def main():
    for ext in ("png", "pdf"):
        if os.path.exists(f"{OUT}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {OUT}.{ext}")
    b20 = next(b for b in json.load(open(os.path.join(
        REPO, "results", "task1_abstention_K20_50.json")))["blocks"] if b["K"] == 20)
    b200 = json.load(open(os.path.join(
        REPO, "results", "task1_abstention_K200.json")))["blocks"][0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    panel(axes[0], b20, f"CUB K=20 (n={b20['n']}) — bar cleared")
    panel(axes[1], b200, f"CUB K=200 (n={b200['n']}) — bar cleared at 0.14 only")
    axes[0].set_ylabel("premise support among issued")
    axes[0].legend(frameon=False, fontsize=6.2, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
