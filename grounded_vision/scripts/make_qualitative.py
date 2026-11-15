#!/usr/bin/env python3
"""Full-width qualitative figure: tree depth x explanations (advisor layout).

Rows (fixed selection rules, tie-break = middle sid at median length):
  (a) typical failure: median-length FAILING default (depth-12) reason
  (b) the trade-off: depth-2 reason fully true but prediction WRONG,
      default prediction correct
  (c) a success: all-supported default reason REQUIRED to cite >=1
      positive literal (35/36 do at K=20 -- verified in the dump)
Cells: reasons at depths 2/4/6/12; literals marked supported (green),
refuted (red), part-not-visible (grey ring). Footer per cell: prediction
correct?, |R|, all-true?. On the image: parts cited by the DEFAULT
reason boxed, coloured by their worst status.

Sources: results/ablation_depth_K20_perinstance.json (+ CUB images/boxes).
Output: results/figures/fig_qualitative.{png,pdf}.
"""
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from grounded_vision.data.cub import CUBDataset  # noqa: E402

DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
OUT = os.path.join(REPO, "results", "figures", "fig_qualitative")
GREEN, RED, GREY, INK = "#1a8f3c", "#c0392b", "#8a8a84", "#222"
DEPTHS = ["2", "4", "6", "12"]


def load_instances():
    return json.load(open(os.path.join(
        REPO, "results", "ablation_depth_K20_perinstance.json")))["instances"]


def select_rows(inst):
    d = "12"
    fail = [r for r in inst if not r["by_depth"][d]["all_supported"]]
    med = statistics.median(len(r["by_depth"][d]["literals"]) for r in fail)
    same = sorted((r for r in fail if len(r["by_depth"][d]["literals"]) == med),
                  key=lambda r: r["sid"])
    a = same[len(same) // 2]
    b_cand = [r for r in inst if r["by_depth"]["2"]["all_supported"]
              and not r["by_depth"]["2"]["correct"] and r["by_depth"][d]["correct"]]
    medb = statistics.median(len(r["by_depth"]["2"]["literals"]) for r in b_cand)
    sameb = sorted((r for r in b_cand if len(r["by_depth"]["2"]["literals"]) == medb),
                   key=lambda r: r["sid"])
    b = sameb[len(sameb) // 2]
    succ = [r for r in inst if r["by_depth"][d]["all_supported"]
            and any(l["b"] == 1 for l in r["by_depth"][d]["literals"])]
    medc = statistics.median(len(r["by_depth"][d]["literals"]) for r in succ)
    samec = sorted((r for r in succ if len(r["by_depth"][d]["literals"]) == medc),
                   key=lambda r: r["sid"])
    c = samec[len(samec) // 2]
    return {"(a) typical failure": a, "(b) depth trade-off": b, "(c) a success": c}


def lit_mark(l):
    if not l["part_visible"]:
        return "\u2218", GREY            # part not visible
    return ("\u2713", GREEN) if l["supported"] else ("\u2717", RED)


def worst_color(lits):
    for l in lits:
        if l["part_visible"] and not l["supported"]:
            return RED
    for l in lits:
        if not l["part_visible"]:
            return GREY
    return GREEN


def short(f):
    part, attr = f.split(":", 1)
    name, _, val = attr.rpartition("=")
    return f"{val}\u2011{name.split('_')[0]}\u2011{part[:4]}"


def main():
    if os.path.exists(f"{OUT}.pdf"):
        raise SystemExit(f"REFUSING to overwrite {OUT}.pdf")
    inst = load_instances()
    rows = select_rows(inst)
    ds = CUBDataset(data_dir=DATA, split="test", use_mock_if_missing=False)
    id2path = dict(l.strip().split(" ", 1)
                   for l in open(os.path.join(DATA, "images.txt")))

    fig = plt.figure(figsize=(7.1, 5.4))
    gs = fig.add_gridspec(3, 5, width_ratios=[0.75, 1, 1, 1, 1],
                          hspace=0.55, wspace=0.3)
    for i, (title, rec) in enumerate(rows.items()):
        ax = fig.add_subplot(gs[i, 0])
        k = next(j for j in range(len(ds)) if ds.meta(j)[0] == rec["sid"])
        _, boxes, _, _ = ds.meta(k)
        img = plt.imread(os.path.join(DATA, "images", id2path[str(rec["sid"])]))
        H, W = img.shape[:2]
        ax.imshow(img)
        ax.axis("off")
        by_part = {}
        for l in rec["by_depth"]["12"]["literals"]:
            by_part.setdefault(l["f"].split(":")[0], []).append(l)
        for part, lits in by_part.items():
            if part not in boxes:
                continue
            x, y, w, h = boxes[part]
            col = worst_color(lits)
            ax.add_patch(Rectangle((x * W, y * H), w * W, h * H, fill=False,
                                   ec=col, ls="--" if col == RED else "-",
                                   lw=1.4 if col != GREEN else 1.0))
        species = id2path[str(rec["sid"])].split("/")[0].split(".", 1)[1].replace("_", " ")
        ax.set_title(f"{title}\n{species} (#{rec['sid']})", fontsize=6.4, loc="left")

        for j, dep in enumerate(DEPTHS):
            cell = rec["by_depth"][dep]
            axc = fig.add_subplot(gs[i, j + 1])
            axc.axis("off")
            axc.set_xlim(0, 1); axc.set_ylim(0, 1)
            if j == 0:
                axc.text(-0.28, 0.5, f"true class {rec['true']}",
                         rotation=90, va="center", fontsize=5.5, color=GREY)
            y = 0.97
            for l in cell["literals"][:11]:
                m, col = lit_mark(l)
                axc.text(0.02, y, m, fontsize=6.5, color=col, va="top", weight="bold")
                axc.text(0.14, y, short(l["f"]) + ("" if l["b"] else " \u00b7neg"),
                         fontsize=5.6, color=INK, va="top",
                         family="DejaVu Sans")
                y -= 0.085
            ok = cell["correct"]
            allt = cell["all_supported"]
            axc.text(0.0, -0.10, f"pred {'\u2713' if ok else '\u2717'}  "
                     f"|R|={len(cell['literals'])}  "
                     f"{'all \u2713' if allt else 'not all \u2713'}",
                     fontsize=6.2, color=GREEN if (ok and allt) else
                     (RED if not allt else GREY), va="bottom")
            axc.add_patch(Rectangle((0, 0), 1, 1, fill=False,
                                    ec=GREEN if allt else (RED if not allt else GREY),
                                    lw=0.8, alpha=0.6, transform=axc.transAxes))
            if i == 0:
                axc.set_title(f"depth \u2264 {dep}" + (" (default)" if dep == "12" else ""),
                              fontsize=7)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT}.pdf | rows {[r['sid'] for r in rows.values()]}")


if __name__ == "__main__":
    main()
