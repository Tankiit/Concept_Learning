#!/usr/bin/env python3
"""fig:scale — support vs accuracy vs independence null across K (Setting T).

Red solid: P(all premises true) with Wilson 95% bands. Red hollow: the
independence null of Eq. (eq:null) — support expected if premise errors
were independent given the literal. Grey: tree accuracy. Under each K:
audited n and mean reason length |R|.
Sources: cbm_cem_audit.json, cbm_cem_audit_K200.json, task1_cbcem_K20_50.json,
independence_gap_cub.json (null K=4/10/200), independence_gap_K20_50.json (20/50).

Output: results/figures/fig_scale.{png,pdf} (refuses to overwrite).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "figures", "fig_scale")

# K, n, acc, support, wilson95, null, |R|
ROWS = [
    (4, 118, 0.602, 0.254, (0.184, 0.340), 0.254, 3.6),
    (10, 243, 0.444, 0.115, (0.081, 0.161), 0.076, 5.9),
    (20, 515, 0.416, 0.070, (0.051, 0.095), 0.061, 7.1),
    (50, 600, 0.303, 0.035, (0.023, 0.053), 0.034, 8.6),
    (200, 600, 0.193, 0.017, (0.009, 0.031), 0.013, 10.8),
]


def main():
    for ext in ("png", "pdf"):
        if os.path.exists(f"{OUT}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {OUT}.{ext}")
    K = [r[0] for r in ROWS]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(K, [r[2] for r in ROWS], "o-", color="#777", ms=3, label="prediction correct")
    ax.plot(K, [r[3] for r in ROWS], "o-", color="#c0392b", ms=3,
            label=r"all premises true")
    ax.fill_between(K, [r[4][0] for r in ROWS], [r[4][1] for r in ROWS],
                    color="#c0392b", alpha=0.18, lw=0)
    ax.plot(K, [r[5] for r in ROWS], "o", mfc="none", mec="#c0392b", ms=3.5,
            mew=1.0, ls=":", lw=0.9, label="independence null")
    for x, y, null in zip(K, [r[3] for r in ROWS], [r[5] for r in ROWS]):
        ax.annotate("", xy=(x, null), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#c0392b", alpha=0.35))
    for (x, _, _, _, _, _, ln), (_, nn, *_rest) in zip(ROWS, ROWS):
        ax.annotate(f"$n{{=}}{nn}$\n$|R|{{=}}{ln:.1f}$", (x, 0.0),
                    ha="center", va="bottom", fontsize=5.2, color="#555",
                    annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlim(3, 330)
    ax.set_ylim(0, 0.68)
    ax.set_xlabel("classes ($K$)")
    ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.subplots_adjust(bottom=0.30)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
