#!/usr/bin/env python3
"""Figure 1 — the gap: one certified-wrong example + the survival curve.

Panel (a): one representative failing CUB instance (deterministic pick:
among K=4 failing reasons that cite bill_color=yellow present — the
carrier literal of 88/88 failures — the median-reason-length instance,
middle sid). Part boxes styled by premise truth: green solid =
certified-and-true, red dashed = certified-but-false.
Panel (b): prediction accuracy vs P(all premises true) across
K = 4/10/20/50/200 with Wilson bands and per-point n; the dotted line at
1.0 marks what the certificate itself proves (exactness, by construction).

Numbers come only from committed JSONs:
  cbm_cem_audit.json, cbm_cem_audit_K200.json, task1_cbcem_K20_50.json.
Output: results/figures/fig1_gap.{png,pdf} (refuses to overwrite).

Run from grounded_vision/ :  PYTHONPATH=src python scripts/plot_fig1_gap.py
"""
import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from grounded_vision.data.cub import CUBDataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (K, n, accuracy, P(all|R), wilson95) — provenance: the three audit JSONs
SCALING = [
    (4, 118, 0.60, 0.254, (0.184, 0.340)),
    (10, 243, 0.44, 0.115, (0.081, 0.161)),
    (20, 515, 0.42, 0.070, (0.051, 0.095)),
    (50, 600, 0.30, 0.035, (0.023, 0.053)),
    (200, 600, 0.19, 0.017, (0.009, 0.031)),
]


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
    return same[len(same) // 2], len(fail), len(pi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011")
    ap.add_argument("--audit", default=os.path.join(ROOT, "results", "cbm_cem_audit.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figures", "fig1_gap"))
    args = ap.parse_args()
    for ext in ("png", "pdf"):
        if os.path.exists(f"{args.out}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {args.out}.{ext}")

    rec, n_fail, n_tot = pick_example(args.audit)
    sid = rec["sid"]
    ds = CUBDataset(data_dir=args.data_dir, split="test", use_mock_if_missing=False)
    k = next(i for i in range(len(ds)) if ds.meta(i)[0] == sid)
    _, boxes, _, _ = ds.meta(k)
    id2path = dict(l.strip().split(" ", 1)
                   for l in open(os.path.join(args.data_dir, "images.txt")))
    img_path = os.path.join(args.data_dir, "images", id2path[str(sid)])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    img = plt.imread(img_path)
    H, W = img.shape[:2]
    ax.imshow(img)
    ax.axis("off")
    seen = set()
    for l in rec["literals"]:  # one box per part; a false premise wins the style
        part = l["feature"].split(":")[0]
        if part in seen:
            continue
        seen.add(part)
        x, y, w, h = boxes[part]
        col, ls, lw = (("#1a8f3c", "-", 1.6) if l["supported"]
                       else ("#c0392b", "--", 2.2))
        ax.add_patch(Rectangle((x * W, y * H), w * W, h * H,
                               fill=False, ec=col, ls=ls, lw=lw))

    ax = axes[1]
    K = [r[0] for r in SCALING]
    acc = [r[2] for r in SCALING]
    sup = [r[3] for r in SCALING]
    lo, hi = [r[4][0] for r in SCALING], [r[4][1] for r in SCALING]
    ax.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.8)
    ax.text(K[-1], 1.04, "certificate exact (by construction)",
            ha="right", fontsize=6.5, color="grey")
    ax.plot(K, acc, "o-", color="#777", ms=3.5,
            label="prediction correct (tree accuracy)")
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

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=220)
    print(f"saved {args.out}.png/.pdf  (example sid {sid})")


if __name__ == "__main__":
    main()
