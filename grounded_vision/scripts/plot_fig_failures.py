#!/usr/bin/env python3
"""fig:failures -- (a) what failure looks like, (b) where it concentrates.

Layout: [ beak crop | tail crop ]
        [ replication cards | concentration curves ]

Crops: sid 44 (Fig-1's deterministic example). Beak crop: certified
bill_colour=yellow; measured yellow-pixel share via HSV (bed average over
firing beaks is 3.3%). Tail crop: certified tail_colour=black, refuted.
Cards: CelebA / BDD-OIA top carriers from committed aggregates (raw images
not local). Curves: share of failing reasons containing at least one of
the top-k carrier literals -- CUB K=4, five re-draws (thin), CelebA.
BDD-OIA has no committed per-instance records; card states its carrier.

Sources: cbm_cem_audit.json, task2_draw*.json, celeba_audit_smiling.json,
bdd_oia_audit.json. Output: results/figures/fig_failures.{png,pdf}.
"""
import glob
import json
import os
import statistics
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from grounded_vision.data.cub import CUBDataset  # noqa: E402

DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
OUT = os.path.join(REPO, "results", "figures", "fig_failures")
RED, GREY = "#c0392b", "#556"


def share_curve(records, kmax=8):
    failing = [r for r in records if r["literals"] and not r["all_supported"]]
    cnt = Counter()
    for r in failing:
        for key in {(l["feature"], l["bit"]) for l in r["literals"]}:
            cnt[key] += 1
    top = [k for k, _ in cnt.most_common(kmax)]
    shares = [sum(1 for r in failing
                  if any((l["feature"], l["bit"]) in set(top[:k]) for l in r["literals"]))
              / len(failing) for k in range(1, kmax + 1)]
    return shares, len(failing)


def yellow_pct(crop):
    rgb = crop.reshape(-1, 3).astype(float) / 255.0
    mx, mn = rgb.max(1), rgb.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    val = mx
    d_ = mx - mn
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hue = np.zeros_like(mx)
    m = d_ > 1e-6
    hue[m] = np.select([rgb[m, 0] == mx[m], rgb[m, 1] == mx[m]],
                       [((g - b)[m] / d_[m]) % 6, ((b - r)[m] / d_[m]) + 2],
                       default=((r - g)[m] / d_[m]) + 4) * 40
    return 100 * np.mean((hue > 15) & (hue < 70) & (sat > 0.35) & (val > 0.30))


def main():
    for ext in ("png", "pdf"):
        if os.path.exists(f"{OUT}.{ext}"):
            raise SystemExit(f"REFUSING to overwrite {OUT}.{ext}")
    d = json.load(open(os.path.join(REPO, "results", "cbm_cem_audit.json")))
    blk = next(b for b in d["blocks"] if b["setting"] == "T_tree_head" and b["K"] == 4)
    pi = [r for r in blk["per_instance"] if r["literals"]]
    fail = [r for r in pi if not r["all_supported"]]
    carriers = [r for r in fail if any(l["feature"] == "beak:bill_color=yellow"
                                       and l["bit"] == 1 for l in r["literals"])]
    med = statistics.median(len(r["literals"]) for r in carriers)
    same = sorted((r for r in carriers if len(r["literals"]) == med), key=lambda r: r["sid"])
    rec = same[len(same) // 2]
    sid = rec["sid"]
    ds = CUBDataset(data_dir=DATA, split="test", use_mock_if_missing=False)
    k = next(i for i in range(len(ds)) if ds.meta(i)[0] == sid)
    _, boxes, _, _ = ds.meta(k)
    id2path = dict(l.strip().split(" ", 1) for l in open(os.path.join(DATA, "images.txt")))
    img = plt.imread(os.path.join(DATA, "images", id2path[str(sid)]))
    H, W = img.shape[:2]

    fig = plt.figure(figsize=(8.8, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.5, wspace=0.3)

    def show_crop(part, ax):
        x, y, w, h = boxes[part]
        x0, y0 = max(0, int(x * W)), max(0, int(y * H))
        x1, y1 = min(W, int((x + w) * W)), min(H, int((y + h) * H))
        crop = img[y0:y1, x0:x1]
        ax.imshow(crop)
        ax.add_patch(Rectangle((0, 0), crop.shape[1] - 1, crop.shape[0] - 1,
                               fill=False, ec=RED, ls="--", lw=1.8))
        ax.axis("off")
        return crop

    ax00 = fig.add_subplot(gs[0, 0])
    beak = show_crop("beak", ax00)
    ax00.set_title("beak crop — certified bill_colour = yellow\n"
                   f"({yellow_pct(beak):.0f}% yellow pixels here; bed average "
                   "over firing beaks 3%; annotated: not yellow)",
                   fontsize=6.4, color=RED)
    ax01 = fig.add_subplot(gs[0, 1])
    show_crop("tail", ax01)
    ax01.set_title("tail crop — certified tail_colour = black\n"
                   "(annotated: not black)", fontsize=6.4, color=RED)

    # ---- CelebA: real face selected by annotation ---------------------------
    cele = json.load(open(os.path.join(REPO, "results", "celeba_audit_smiling.json")))
    fails_c = [r for r in cele["per_instance"] if not r["all_true"]]
    cc = Counter()
    for r in fails_c:
        for lit, b in r["reason"]:
            cc[(lit, b)] += 1
    (lit, b), n_top = cc.most_common(1)[0]
    cdata = "/Users/tanmoy/research/data/celeba"
    part = [l.split() for l in open(f"{cdata}/list_eval_partition.txt").read().strip().splitlines()]
    alines = open(f"{cdata}/list_attr_celeba.txt").read().strip().splitlines()
    cattrs = alines[1].split()
    hc = cattrs.index("High_Cheekbones")
    rows = [l.split() for l in alines[2:]]
    attrmap = {r[0]: [int(x) for x in r[1:]] for r in rows}
    # annotation-picked (audit export kept no filenames): middle test-partition
    # image whose annotations AFFIRM the cheekbones the carrier literal denies
    cands = sorted(f for f, p in part if p == "2" and attrmap[f][hc] > 0)
    cfile = cands[len(cands) // 2]
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(plt.imread(f"{cdata}/img_align_celeba/{cfile}"))
    ax10.axis("off")
    ax10.set_title(f"CelebA — carrier literal denies what this face affirms\n"
                   f"certified {lit} = {b} in {n_top / len(fails_c):.0%} of "
                   f"{len(fails_c):,} failing reasons;\nannotations: "
                   f"{lit} present (image selected by annotation)",
                   fontsize=6.0, color=GREY)

    # ---- BDD-OIA card (no local images) --------------------------------------
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.axis("off")
    ax02.set_title("BDD-OIA — top carrier", fontsize=7, color=GREY)
    bdd = json.load(open(os.path.join(REPO, "results", "bdd_oia_audit.json")))
    gl = bdd["literal_marginals"]["green_light=1"]
    ax02.text(0.5, 0.52, "top fault slot: green_light = 1\n"
              f"(support {gl['support']:.1%}, {gl['n']:,} occurrences)\n\n"
              "most stereotyped: rider = 0\nin 76% of failing reasons\n(support 0.92)\n\n"
              "pick frame 6c0cc8ca-e1dd9997_3\n(bdd100k frames not local)",
              ha="center", va="center", fontsize=6.2, color=GREY,
              transform=ax02.transAxes)
    ax02.add_patch(Rectangle((0.06, 0.18), 0.88, 0.64, fill=False, ec="#bbb", lw=0.8,
                             transform=ax02.transAxes))

    # ---- concentration curves ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1:])
    main_curve, n_fail = share_curve(pi)
    ks = np.arange(1, len(main_curve) + 1)
    ax.plot(ks, main_curve, "o-", color=RED, ms=3.5, lw=1.4,
            label=f"CUB K=4 ({n_fail} failing)")
    for f in sorted(glob.glob(os.path.join(REPO, "results", "task2_draw*.json"))):
        dd = json.load(open(f))
        b2 = next(b for b in dd["blocks"] if b["setting"] == "T_tree_head")
        cur, _ = share_curve(b2["per_instance"])
        ax.plot(ks[:len(cur)], cur, "-", color=RED, lw=0.6, alpha=0.35)
    cele_rec = [{"literals": [{"feature": nm, "bit": b} for nm, b in r["reason"]],
                 "all_supported": r["all_true"]} for r in cele["per_instance"]]
    cele_rec = [r for r in cele_rec if r["literals"]]
    cur_c, nfc = share_curve(cele_rec)
    ax.plot(ks[:len(cur_c)], cur_c, "s-", color=GREY, ms=3.5, lw=1.4,
            label=f"CelebA ({nfc:,} failing)")
    bdd_rec = [{"literals": [{"feature": l[0], "bit": l[1]} for l in r["reason"]],
                "all_supported": r["all_true"]} for r in bdd["reasons"]]
    cur_b, nfb = share_curve(bdd_rec)
    ax.plot(ks[:len(cur_b)], cur_b, "^--", color="#2a6", ms=3.5, lw=1.4,
            label=f"BDD-OIA ({nfb:,} failing)")
    ax.set_xlabel("top-$k$ carrier literals")
    ax.set_ylabel("share of failing reasons")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("concentration: thin lines = five CUB re-draws", fontsize=7)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
