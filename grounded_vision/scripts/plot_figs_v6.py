#!/usr/bin/env python3
"""Camera-ready experiment figures (2026-11-15), styled per the advisor's
script; numbers are LOADED from the committed JSONs (the original script's
TODO), each figure naming its source below.

Outputs to results/figures/: fig_scale, fig_abstain (restyled revisions),
fig_ceiling, fig_methods, fig_readers (new). Refuses to overwrite.
Run: python scripts/plot_figs_v6.py   (from grounded_vision/)
"""
import json
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
                    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                    "ytick.major.width": 0.6, "axes.edgecolor": "#555", "xtick.color": "#333",
                    "ytick.color": "#333", "axes.labelcolor": "#222", "pdf.fonttype": 42})
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a94", "#222"
W = 3.3  # CVPR column width, inches

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: os.path.join(os.path.dirname(HERE), "results", name)
OUT = os.path.join(os.path.dirname(HERE), "results", "figures")


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        if os.path.exists(p):
            raise SystemExit(f"REFUSING to overwrite {p}")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200, bbox_inches="tight")


def jload(name):
    return json.load(open(R(name)))


# ================= Fig: scale ==============================================
# cbm_cem_audit.json + task1_cbcem_K20_50.json + cbm_cem_audit_K200.json
# (acc, support, |R|); independence_gap_cub.json + _K20_50.json (baseline).
aud = jload("cbm_cem_audit.json")
t1 = jload("task1_cbcem_K20_50.json")
aud200 = jload("cbar" if False else "cbm_cem_audit_K200.json")
blocks = {b["K"]: b for b in aud["blocks"] if b["setting"] == "T_tree_head"}
blocks.update({b["K"]: b for b in t1["blocks"] if b["setting"] == "T_tree_head"})
blocks[200] = next(b for b in aud200["blocks"] if b["setting"] == "T_tree_head")
ig = jload("independence_gap_cub.json")
igg = {b["K"]: b for b in ig["blocks"] if b["setting"].startswith("T")}
ig20_50 = jload("independence_gap_K20_50.json")["blocks"]

K, n, acc, sup, sci, nul, Rl = [], [], [], [], [], [], []
for k in (4, 10, 20, 50, 200):
    a = blocks[k]["audit"]
    K.append(k); n.append(blocks[k]["n_audited"]); acc.append(a["accuracy"])
    sup.append(a["p_all_supported"]); sci.append(tuple(a["p_all_supported_wilson95"]))
    Rl.append(a["mean_len"])
    if k in igg:
        nul.append(igg[k]["independence_predicted_joint"])
    else:
        nul.append(ig20_50[f"K{k}"]["null"])
print("fig_scale loaded:", list(zip(K, acc, sup, nul)))

fig, ax = plt.subplots(figsize=(W, 2.3))
x = range(len(K))
alo, ahi = zip(*[wilson(a, m) for a, m in zip(acc, n)])
ax.fill_between(x, alo, ahi, color=BLUE, alpha=.15, lw=0)
ax.plot(x, acc, "-o", color=BLUE, lw=2, ms=4)
slo, shi = zip(*sci)
ax.fill_between(x, slo, shi, color=ORANGE, alpha=.15, lw=0)
ax.plot(x, sup, "-o", color=ORANGE, lw=2, ms=4)
ax.plot(x, nul, "o", mfc="white", mec=INK, mew=1, ms=5, ls="none")
ax.set_yscale("log"); ax.set_ylim(.007, 1)
ax.set_yticks([.01, .03, .1, .3, 1]); ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
ax.set_xticks(list(x)); ax.set_xticklabels([f"{k}\n|R|={r:.1f}" for k, r in zip(K, Rl)])
ax.set_xlabel("number of classes $K$"); ax.set_ylabel("probability")
ax.text(4.1, acc[-1], "accuracy", color=INK, va="center", fontsize=7)
ax.text(4.1, sup[-1] * 1.15, "premise\nsupport", color=INK, va="center", fontsize=7)
ax.text(0.05, .011, "○ independence baseline", color="#555", fontsize=7)
ax.grid(axis="y", color="#e6e6e3", lw=.5); ax.set_axisbelow(True)
save(fig, "fig_scale")

# ================= Fig: gate ===============================================
# (a) abstention_curves.json D/10 @0.136; (b) task1_abstention_K*.json
fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * W, 2.1), gridspec_kw={"width_ratios": [1, 1.4]})
cur = jload("abstention_curves.json")
d10 = next(b for b in cur["blocks"] if b["setting"] == "D_distilled" and b["K"] == 10)
cell = d10["at_coverage"]["0.136"]
ch = ["random", "raw conf.", "calibrated", "oracle"]
v = [cell["D_random"]["joint"], cell["C"]["joint"], cell["B"]["joint"], cell["A_ir"]["joint"]]
blo, bhi = cell["B"]["boot_pct"]
col = [GRAY, YELLOW, ORANGE, "#bbb"]
a1.bar(range(4), v, color=col, width=.6)
a1.errorbar(2, v[2], yerr=[[v[2] - blo], [bhi - v[2]]], color=INK, lw=1, capsize=2)
for i, val in enumerate(v):
    a1.text(i, val + .03, f"{val:.2f}", ha="center", fontsize=7, color=INK)
a1.set_xticks(range(4)); a1.set_xticklabels(ch, fontsize=7)
a1.set_ylabel("support among issued reasons"); a1.set_ylim(0, 1)
a1.set_title("(a) $K{=}10$, Setting D, coverage 0.14", fontsize=8, loc="left")

t20_50 = {b["K"]: b for b in jload("task1_abstention_K20_50.json")["blocks"]}
t200 = jload("task1_abstention_K200.json")["blocks"][0]
rows = [(20, "0.140", t20_50[20]), (50, "0.140", t20_50[50]),
        (200, "0.140", t200), (200, "0.100", t200), (200, "0.050", t200)]
for i, (k, cov, blk) in enumerate(rows):
    c = blk["at_coverage"][cov]["B"]
    base, s, lo, hi = blk["raw_joint"], c["joint"], *c["boot_pct"]
    a2.plot([i - .3, i + .3], [2 * base] * 2, color=INK, ls="--", lw=1)
    a2.plot(i, base, "o", mfc="white", mec=GRAY, mew=1.2, ms=5)
    a2.errorbar(i, s, yerr=[[s - lo], [hi - s]], fmt="o", color=ORANGE, ms=5, lw=1.2, capsize=2)
a2.set_xticks(range(5))
a2.set_xticklabels([f"K={k}\ncov {float(c):.2f}" for k, c, _ in rows], fontsize=7)
a2.set_ylabel("premise support"); a2.set_ylim(0, .5)
a2.set_title("(b) Setting T: issued (●) vs. all (○); dashed = 2× bar", fontsize=8, loc="left")
a2.grid(axis="y", color="#e6e6e3", lw=.5); a2.set_axisbelow(True)
fig.tight_layout(); save(fig, "fig_abstain")

# ================= Fig: ceiling ============================================
# design_b_v2.json ladder (max over depth per input); BDD stage6 forward.
db = jload("design_b_v2.json")["ladder"]
def lad(key):
    return max(r["fidelity"] for r in db[key]["rows"])
cub = [lad("bits36"), lad("region384"), lad("scores36"), lad("gtbits36")]
fw = jload("bdd_oia_experiments.json")["stage6_ladder"]["forward"]
bdd = [fw["deployed_tree_acc"], fw["direct_probe_acc"], None, fw["gtbits_teacher_acc"]]
print("fig_ceiling loaded:", cub, bdd)
stages = ["predicted\nbits", "region\nfeatures", "continuous\nscores", "ground-truth\nbits"]
fig, ax = plt.subplots(figsize=(W, 2.1))
ax.plot(range(4), cub, "-o", color=BLUE, lw=2, ms=4)
ax.plot([0, 1], bdd[:2], "-o", color=AQUA, lw=2, ms=4)
ax.plot([1, 3], [bdd[1], bdd[3]], "--", color=AQUA, lw=1.5)
ax.plot(3, bdd[3], "o", color=AQUA, ms=4)
ax.text(2, .82, "not run", ha="center", fontsize=6.5, color="#777")
ax.text(3.08, cub[-1], "CUB ($K{=}4$)", va="center", fontsize=7, color=INK)
ax.text(3.08, bdd[-1], "BDD-OIA\n(forward)", va="center", fontsize=7, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=7)
ax.set_ylabel("head accuracy"); ax.set_ylim(.55, 1.02); ax.set_xlim(-.2, 3.9)
ax.grid(axis="y", color="#e6e6e3", lw=.5); ax.set_axisbelow(True)
save(fig, "fig_ceiling")

# ================= Fig: methods (sufficiency vs support) ===================
# bdd_oia_methods.json part1 (forces_rate = sufficiency, p_all = support)
mm = jload("bdd_oia_methods.json")["part1_explanation_methods"]["methods"]
meth = [("minimal sufficient\nreason", mm["ours"]["forces_rate"], mm["ours"]["p_all"]),
        ("greedy erasure", mm["erasure"]["forces_rate"], mm["erasure"]["p_all"]),
        ("top-3 importance", mm["topk"]["forces_rate"], mm["topk"]["p_all"]),
        ("random-3", mm["random"]["forces_rate"], mm["random"]["p_all"])]
fig, ax = plt.subplots(figsize=(W, 2.0))
for name, s, p in meth:
    c = ORANGE if s == 1 else GRAY
    ax.plot(s, p, "o", color=c, ms=7)
    ax.annotate(name, (s, p), textcoords="offset points",
                xytext=(6, 4) if s < .9 else (-6, 6),
                ha="left" if s < .9 else "right", fontsize=7, color=INK)
ax.set_xlabel("sufficiency for the head"); ax.set_ylabel("premise support")
ax.set_xlim(-.08, 1.1); ax.set_ylim(.5, .75)
ax.grid(color="#e6e6e3", lw=.5); ax.set_axisbelow(True)
save(fig, "fig_methods")

# ================= Fig: reader strength vs gap =============================
# reader acc: cbm_cem_audit concept_quality / bdd + celeba audits / sun;
# gap = teacher acc - premise support, per bed (K=4 for CUB/SUN).
ca = aud["concept_quality_summary"]["mean_accuracy"]
bd = jload("bdd_oia_audit.json")
ACTIVE = ("forward", "stop", "left", "right")
bd_acc = sum(bd["per_action"][a]["teacher_acc"] for a in ACTIVE) / 4
bd_read = sum(bd["reader_valid_acc"]) / len(bd["reader_valid_acc"])
ce = jload("celeba_audit_smiling.json")
sun = jload("sun_audit.json")["K4"]
pts = [("CUB", ca, blocks[4]["audit"]["accuracy"] - sup[0]),
       ("BDD-OIA", bd_read, bd_acc - bd["pooled_premise_support"]["p"]),
       ("CelebA", ce["reader_valid_acc_mean"],
        ce["head"]["teacher_acc"] - ce["premise_support"]["p"]),
       ("SUN$^*$", sun["reader_valid_acc_mean"],
        sun["teacher_acc"] - sun["premise_support"]["p"])]
print("fig_readers loaded:", [(a, round(r, 3), round(g * 100, 1)) for a, r, g in pts])
fig, ax = plt.subplots(figsize=(W, 2.0))
off = {"CUB": (5, 4), "BDD-OIA": (-8, 6), "CelebA": (6, -10), "SUN$^*$": (5, 4)}
for name, r_, g in pts:
    ax.plot(r_, g * 100, "o", color=BLUE, ms=6)
    dx, dy = off[name]
    ax.annotate(name, (r_, g * 100), textcoords="offset points", xytext=(dx, dy),
                fontsize=7, color=INK, ha="right" if dx < 0 else "left")
ax.set_xlabel("concept-reader accuracy"); ax.set_ylabel("accuracy − support (pts)")
ax.set_xlim(.68, 1.0); ax.set_ylim(0, 40)
ax.grid(color="#e6e6e3", lw=.5); ax.set_axisbelow(True)
save(fig, "fig_readers")
print("ok")
