#!/usr/bin/env python3
"""Method figure (fig:pipeline): pipeline row + the audit rungs beneath it.

Top row : image (oracle part boxes) -> predicates -> Boolean tree head ->
          certified reason.
Rungs   : proof (exact) / fidelity (measured) / premise support (measured),
          then a dashed separator and the GATE row (the lever, not a rung).
The K-curve is NOT inlined here -- it is Fig. 1(b); rung 3's cell carries
the endpoints 0.254 -> 0.017 only.
All numbers from committed JSONs (cbm_cem_audit*, conditional_fidelity,
pyxai cross-check ledger in REVIEW_VALIDATION_ROADMAP / app:pyxai).

Output: results/figures/fig_pipeline.{png,pdf} (refuses to overwrite).
Run from grounded_vision/ : PYTHONPATH not required (matplotlib only).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "figures", "fig_pipeline")

PIPE = [
    ("image", "oracle part boxes"),
    ("predicates $\\phi(x)$", "36 bound part–attribute literals\nlogistic readers, val-isotonic"),
    ("Boolean tree head", "CART depth 12, $K$ classes\nSetting T / Setting D"),
    ("certified reason $R$", "$R \\vDash c$, minimal, exact\nengine cross-check 472/472"),
]
RUNGS = [
    ("rung 1 · proof", "$R$ sufficient for the tree", "exact",
     "by construction; independent engine agrees 472/472", "#1a8f3c", "✓"),
    ("rung 2 · fidelity", "tree $\\leftrightarrow$ neural teacher", "measured",
     "0.81 at 4 classes / 0.61 at 10 — partial, not binding", "#e08a1e", "≈"),
    ("rung 3 · premise support", "cited literals true of the image", "measured",
     "0.254 → 0.017 (4 → 200 classes); always below accuracy — binding", "#c0392b", "✗"),
    ("gate · the lever", "calibrated min-confidence abstention", "deployment",
     "restores support at partial coverage; pays coverage", "#556", "◧"),
]


def main():
    for ext in ("png", "pdf"):
        if os.path.exists(f"{OUT}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {OUT}.{ext}")

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- pipeline row -----------------------------------------------------
    xs = [1.5, 26.5, 51.5, 76.5]
    w, y0, h = 22.0, 84.0, 15.0
    for (title, sub), x in zip(PIPE, xs):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.4",
                                    fc="#f7f7f7", ec="#333", lw=1.0))
        ax.text(x + w / 2, y0 + h - 4.5, title, ha="center", fontsize=8.5, weight="bold")
        ax.text(x + w / 2, y0 + 4.6, sub, ha="center", fontsize=6.4, color="#333")
    for x in (xs[1] - 2.6, xs[2] - 2.6, xs[3] - 2.6):
        ax.add_patch(FancyArrowPatch((x - 0.6, y0 + h / 2), (x + 1.4, y0 + h / 2),
                                     arrowstyle="-|>", mutation_scale=11, color="#333"))

    # ---- audit panel: one rung per row -----------------------------------
    ax.text(1.5, 75.0, "the audit — one rung per row", fontsize=8, style="italic", color="#333")
    row_y = [63.0, 48.5, 34.0, 16.0]
    for (name, claim, kind, evidence, col, glyph), y in zip(RUNGS, row_y):
        if name.startswith("gate"):
            ax.plot([1.5, 98.5], [y + 12.8, y + 12.8], ls="--", lw=0.8, color="#999")
        ax.add_patch(FancyBboxPatch((1.5, y), 24.5, 11.4, boxstyle="round,pad=0.3",
                                    fc="white", ec=col, lw=1.1))
        ax.text(3.2, y + 7.6, name, fontsize=7.6, weight="bold", color=col)
        ax.text(3.2, y + 3.1, claim, fontsize=6.8, color="#333")
        ax.add_patch(FancyBboxPatch((28.0, y), 11.5, 11.4, boxstyle="round,pad=0.3",
                                    fc=col, ec=col, lw=1.1, alpha=0.14))
        ax.text(33.75, y + 5.7, glyph, ha="center", fontsize=12, color=col)
        ax.text(33.75, y + 1.9, kind, ha="center", fontsize=5.8, color=col)
        ax.add_patch(FancyBboxPatch((41.5, y), 57.0, 11.4, boxstyle="round,pad=0.3",
                                    fc="white", ec="#bbb", lw=0.8))
        ax.text(43.2, y + 5.7, evidence, fontsize=6.8, color="#333", va="center")

    ax.text(1.5, 2.5, "the K-curve is Fig. 1(b); rung 3 carries its endpoints only",
            fontsize=6.0, color="#777", style="italic")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
