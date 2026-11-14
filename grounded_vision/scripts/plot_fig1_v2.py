#!/usr/bin/env python3
"""Figure 1 v2 — pipeline strip (measured | exact) over the two panels.

Strip  : the five stages of the audit, split by WHERE THE PROOF STOPS —
         a vertical dashed rule on the arrow between phi(x) and the tree.
         Left (measured): image, DINOv3 patches, readers — learned from
         the image, the only stages a reviewer can break by changing it.
         Right (exact): tree routing, reason extraction + verification —
         unaffected by a wrong premise. phi(x) carries sid 44's five
         reason bits (three false / two true), tying the strip to (a).
Panels : (a) the certified-wrong albatross (unchanged from fig1_gap),
         (b) proved-vs-survives across K (unchanged from fig1_gap).

Numbers only from committed JSONs (cbm_cem_audit.json incl. sid 44,
cbm_cem_audit_K200.json, task1_cbcem_K20_50.json).
Output: results/figures/fig1_v2.{png,pdf} (refuses to overwrite).
Run from grounded_vision/ :  PYTHONPATH=src python scripts/plot_fig1_v2.py
"""
import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

from grounded_vision.data.cub import CUBDataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCALING = [
    (4, 118, 0.60, 0.254, (0.184, 0.340)),
    (10, 243, 0.44, 0.115, (0.081, 0.161)),
    (20, 515, 0.42, 0.070, (0.051, 0.095)),
    (50, 600, 0.30, 0.035, (0.023, 0.053)),
    (200, 600, 0.19, 0.017, (0.009, 0.031)),
]

# (label, sublabel, side); boundary drawn on the arrow after phi(x)
STAGES = [
    ("image", "CUB, oracle part boxes", "measured"),
    ("DINOv3 patches", "frozen", "measured"),
    ("predicates $\\phi(x)$", "$F{=}36$ part-bound literals\nlogistic readers", "measured"),
    ("decision tree", "CART $d{=}12$, predicted bits\nSetting T / distilled D", "exact"),
    ("sufficient reason $R$", "$R \\vDash c$, minimal\nengine cross-check", "exact"),
]
MEAS, EXACT = "measured", "exact"
SIDE_FC = {MEAS: "#f9ede7", EXACT: "#eaf2ea"}
SIDE_EC = {MEAS: "#c0392b", EXACT: "#1a8f3c"}


def pick_example(audit_json):
    d = json.load(open(audit_json))
    blk = next(b for b in d["blocks"]
               if b["setting"] == "T_tree_head" and b["K"] == 4)
    pi = [r for r in blk["per_instance"] if r["literals"]]
    fail = [r for r in pi if not r["all_supported"]]
    carriers = [r for r in fail
                if any(l["feature"] == "beak:bill_color=yellow" and l["bit"] == 1
                       for l in r["literals"])]
    med = statistics.median(len(r["literals"]) for r in carriers)
    same = sorted((r for r in carriers if len(r["literals"]) == med),
                  key=lambda r: r["sid"])
    return same[len(same) // 2]


def panel_pipeline(ax, rec):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    n = len(STAGES)
    w = 16.2
    gap = (100 - n * w) / (n - 1)
    xs = [i * (w + gap) for i in range(n)]
    bnd_x = xs[3] - gap / 2                      # boundary: after phi(x)
    # region tints + bracket labels
    ax.add_patch(Rectangle((0, 6), bnd_x - 1, 88, fc=SIDE_FC[MEAS], ec="none", alpha=0.5))
    ax.add_patch(Rectangle((bnd_x + 1, 6), 100 - bnd_x - 1, 88, fc=SIDE_FC[EXACT], ec="none", alpha=0.5))
    ax.text(bnd_x / 2 - 3, 1.5, "measured — learned from the image", ha="center",
            fontsize=6.4, color=SIDE_EC[MEAS], style="italic")
    ax.text(bnd_x + (100 - bnd_x) / 2 + 3, 1.5, "exact — by construction", ha="center",
            fontsize=6.4, color=SIDE_EC[EXACT], style="italic")
    for (title, sub, side), x in zip(STAGES, xs):
        ax.add_patch(FancyBboxPatch((x, 34), w, 58, boxstyle="round,pad=0.5",
                                    fc="white", ec=SIDE_EC[side], lw=1.2))
        ax.text(x + w / 2, 82, title, ha="center", fontsize=7.6, weight="bold")
        ax.text(x + w / 2, 54, sub, ha="center", fontsize=5.6, color="#333")
    for x in xs[1:]:
        ax.add_patch(FancyArrowPatch((x - gap + 1.2, 63), (x - 1.2, 63),
                                     arrowstyle="-|>", mutation_scale=10, color="#333"))
    # the boundary: where the proof stops
    ax.plot([bnd_x, bnd_x], [12, 96], ls="--", lw=1.4, color="#222")
    ax.annotate("where the proof stops", (bnd_x, 96), ha="center", va="bottom",
                fontsize=6.6, weight="bold")
    # sid 44's reason bits under phi(x): three false, two true
    for i, ok in enumerate((False, False, True, True, False)):
        ax.text(xs[2] + w / 2 - 12.8 + i * 6.4, 24, "\u2717" if not ok else "\u2713",
                ha="center", fontsize=7.5, color=SIDE_EC[MEAS] if not ok else SIDE_EC[EXACT],
                weight="bold")
    ax.text(xs[2] + w / 2, 15.5, "the five reason bits (sid 44)", ha="center",
            fontsize=5.2, color="#777", style="italic")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011")
    ap.add_argument("--audit", default=os.path.join(ROOT, "results", "cbm_cem_audit.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figures", "fig1_v2"))
    args = ap.parse_args()
    for ext in ("png", "pdf"):
        if os.path.exists(f"{args.out}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {args.out}.{ext}")

    rec = pick_example(args.audit)
    sid = rec["sid"]
    ds = CUBDataset(data_dir=args.data_dir, split="test", use_mock_if_missing=False)
    k = next(i for i in range(len(ds)) if ds.meta(i)[0] == sid)
    _, boxes, _, _ = ds.meta(k)
    id2path = dict(l.strip().split(" ", 1)
                   for l in open(os.path.join(args.data_dir, "images.txt")))
    img_path = os.path.join(args.data_dir, "images", id2path[str(sid)])

    fig = plt.figure(figsize=(8.8, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.55, 1], width_ratios=[1.15, 1],
                          hspace=0.42, wspace=0.18)
    panel_pipeline(fig.add_subplot(gs[0, :]), rec)

    # ---- panel (a): the certified-wrong example (as fig1_gap) -------------
    ax = fig.add_subplot(gs[1, 0])
    img = plt.imread(img_path)
    H, W = img.shape[:2]
    ax.imshow(img)
    ax.axis("off")
    seen = set()
    for l in rec["literals"]:
        part = l["feature"].split(":")[0]
        if part in seen:
            continue
        seen.add(part)
        x, y, w, h = boxes[part]
        col, ls, lw = (("#1a8f3c", "-", 1.6) if l["supported"] else ("#c0392b", "--", 2.2))
        ax.add_patch(Rectangle((x * W, y * H), w * W, h * H, fill=False, ec=col, ls=ls, lw=lw))

    # ---- panel (b): proved vs survives across K (as fig1_gap) -------------
    ax = fig.add_subplot(gs[1, 1])
    K = [r[0] for r in SCALING]
    acc = [r[2] for r in SCALING]
    sup = [r[3] for r in SCALING]
    lo, hi = [r[4][0] for r in SCALING], [r[4][1] for r in SCALING]
    ax.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.8)
    ax.text(K[-1], 1.04, "certificate exact (by construction)",
            ha="right", fontsize=6.5, color="grey")
    ax.plot(K, acc, "o-", color="#777", ms=3.5, label="prediction correct (tree accuracy)")
    ax.plot(K, sup, "o-", color="#c0392b", ms=3.5,
            label=r"all cited premises true  $P(\mathrm{all}\,|\,R)$")
    ax.fill_between(K, lo, hi, color="#c0392b", alpha=0.18, lw=0)
    for (x, y, _, _, _), (_, nn, _, _, _) in zip(SCALING, SCALING):
        ax.annotate(f"n={nn}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=6, color="#c0392b")
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.set_xticklabels([str(v) for v in K])
    ax.set_xlim(3, 320)
    ax.set_xlabel("classes (K)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=6.8, loc="center right")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {args.out}.png/.pdf  (example sid {sid})")


if __name__ == "__main__":
    main()
