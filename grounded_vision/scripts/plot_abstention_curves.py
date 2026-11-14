"""Figure: coverage-support abstention curves, CUB (abstention_curves.json)
+ CelebA + BDD-OIA (our audit JSONs). Three panels, B/C/random (+A where
available). Writes tmp/abstention_curves_draft.png (draft for the paper
figure; final version moves into the LaTeX tree).

Env: any with matplotlib + numpy.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "abstention_curves_draft.png"

panels = []


def add_panel(title, rows):
    """rows: list of (label, coverage, p_all)"""
    panels.append((title, rows))


# --- CUB: abstention_curves.json (blocks[].at_coverage[cov][score].joint) ---
ac = json.loads((ROOT / "results" / "abstention_curves.json").read_text())
blk = next((b for b in ac["blocks"]
            if b["setting"] == "D_distilled" and b["K"] == 10), None)
if blk is not None:
    rows = []
    for score, label in [("A_ir", "A (oracle)"), ("A_cq", "A (discrete)"),
                         ("B", "B (calibrated)"), ("C", "C (raw)"),
                         ("D_random", "random")]:
        covs = sorted(float(c) for c in blk["at_coverage"])
        pts = [(c, blk["at_coverage"][f"{c:.3f}"][score]["joint"]) for c in covs]
        rows.append((label, [p[0] for p in pts], [p[1] for p in pts]))
    add_panel("CUB, Setting D, 10 classes", rows)

# --- CelebA / BDD-OIA: our audit JSONs (abstention_curve lists) ---
for fname, title in [("celeba_audit_smiling.json", "CelebA (Smiling)"),
                     ("bdd_oia_audit.json", "BDD-OIA (driving actions)")]:
    d = json.loads((ROOT / "results" / fname).read_text())
    rows = []
    for code, label in [("A_discrete", "A (discrete)"), ("B_cal", "B (calibrated)"),
                        ("C_raw", "C (raw)")]:
        pts = [c for c in d["abstention_curve"] if c["score"] == code]
        if pts:
            rows.append((label, [c["coverage"] for c in pts],
                         [c["p_all_given_cov"] for c in pts]))
    rnd = np.mean([[c["random"]] for c in d["abstention_curve"][:1]], axis=0)[0]
    rows.append(("random", [0, 1], [rnd, rnd]))
    add_panel(title, rows)

fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
colors = {"A (discrete)": "#999", "B (calibrated)": "#1a6faa", "C (raw)": "#c25b1e",
          "random": "#bbb", "B (calibrated min-conf)": "#1a6faa"}
for ax, (title, rows) in zip(axes[0], panels):
    for label, cov, p in rows:
        ax.plot(cov, p, marker="o", ms=3, lw=1.6, label=label,
                color=colors.get(label.split(" (")[0] if "(" in label else label, None))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("coverage")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
axes[0][0].set_ylabel(r"premise support $P(\mathrm{all}\,|\,\mathrm{cov})$")
axes[0][-1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT, dpi=180)
print("wrote", OUT)
