#!/usr/bin/env python3
"""Task 1 (due first): calibrated abstention at K=20, 50 and the calibrated arm at K=200.

Steps:
  1. audit pipeline at K=20,50 (premise support + reasons; writes task1_cbcem_K20_50.json)
  2. X2 pipeline at K=20,50,200 (calibrated/raw literal confidences; task1_x2_K20_50_200.json)
  3. abstention curves per K (joins by sid; K=200 joins the new X2 run to the
     COMMITTED cbm_cem_audit_K200.json per-instance records -- never rewritten)
  4. figure: support vs coverage, one calibrated curve per K (PNG+PDF in results/figures/)
  5. pre-registered verdicts (GRACEFUL / CLIFF / AMBIGUOUS at c=0.14 vs 2x unconditional),
     per docs/wiki/CVPR2027_PREREG_TASK_RUNS.md -- criteria fixed before this run.

Send the printed tables as soon as they exist, even if they look bad.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402


def curve_json(path):
    return json.load(open(path))


def verdict_at(block, c_key="0.140"):
    cell = block["at_coverage"][c_key]
    b = cell["B"]
    r = block["raw_joint"]
    lo, hi = b["boot_pct"]
    lo2, hi2 = 2 * r, 2 * r
    if lo >= lo2:
        return "GRACEFUL", b, r
    if hi < hi2:
        return "CLIFF", b, r
    return "AMBIGUOUS", b, r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=common.DEFAULT_CACHE)
    ap.add_argument("--data-dir", default=common.DEFAULT_DATA)
    ap.add_argument("--audit-ks", default="20,50")
    ap.add_argument("--x2-ks", default="20,50,200")
    ap.add_argument("--x2-extra", default="", help="extra args for the X2 run, e.g. '--B 200'")
    args = ap.parse_args()

    checks = []
    aks = args.audit_ks.replace(",", "_")
    xks = args.x2_ks.replace(",", "_")

    # 1+2: pipelines (new files only)
    audit_out = common.out_path(f"task1_cbcem_K{aks}.json")
    common.run_pipeline("run_cbm_cem_audit.py", audit_out, args.cache_dir, args.data_dir,
                        extra=["--classes", args.audit_ks, "--settings", "T,D",
                               "--seeds", "42"])
    checks.append(common.check(f"audit pipeline wrote {os.path.basename(audit_out)}", True))

    x2_out = common.out_path(f"task1_x2_K{xks}.json")
    extra = ["--classes", args.x2_ks, "--seeds", "42"]
    if args.x2_extra:
        extra += args.x2_extra.split()
    common.run_pipeline("run_x2_uncertainty.py", x2_out, args.cache_dir, args.data_dir,
                        extra=extra)
    checks.append(common.check(f"X2 pipeline wrote {os.path.basename(x2_out)}", True))

    # 3: abstention curves -- K in both files; K=200 vs the committed audit
    cur_a = common.out_path(f"task1_abstention_K{aks}.json")
    common.run_script("abstention_curves.py",
                      ["--x2", x2_out, "--audit", audit_out, "--out", cur_a])
    committed_k200 = os.path.join(common.REPO, "results", "cbm_cem_audit_K200.json")
    cur_b = common.out_path("task1_abstention_K200.json")
    common.run_script("abstention_curves.py",
                      ["--x2", x2_out, "--audit", committed_k200, "--out", cur_b])
    checks.append(common.check("abstention curves written (never overwrote committed files)", True))

    # 4: figure (calibrated arm across K, plus committed K=4/10 for context)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = os.path.join(common.REPO, "results", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        srcs = [os.path.join(common.REPO, "results", "abstention_curves.json"), cur_a, cur_b]
        for path in srcs:
            if not os.path.exists(path):
                continue
            for blk in curve_json(path)["blocks"]:
                if blk["setting"] != "T_tree_head":
                    continue
                name = f"K={blk['K']}"
                cs = sorted(float(c) for c in blk["at_coverage"])
                ys = [blk["at_coverage"][f"{c:.3f}"]["B"]["joint"] for c in cs]
                ax.plot(cs, ys, marker="o", label=name)
                ax.axhline(blk["raw_joint"], color="grey", lw=0.6, alpha=0.4)
        ax.set_xlabel("coverage")
        ax.set_ylabel("joint premise support (calibrated abstention)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(fig_dir, f"task1_abstention_scaling.{ext}"), dpi=200)
        checks.append(common.check("figure results/figures/task1_abstention_scaling.{png,pdf}", True))
    except Exception as e:  # figure is a deliverable; a failure here is a FAIL
        checks.append(common.check(f"figure generation ({e})", False))

    # 5: pre-registered verdicts
    print("\n== Pre-registered verdicts (criteria fixed in CVPR2027_PREREG_TASK_RUNS.md) ==")
    verdicts = {}
    for path in (cur_a, cur_b):
        for blk in curve_json(path)["blocks"]:
            if blk["setting"] != "T_tree_head":
                continue
            key = f"K={blk['K']}"
            if key in verdicts:
                continue
            v, b, r = verdict_at(blk)
            verdicts[key] = v
            print(f"{key}: {v}  B@0.14 = {b['joint']:.3f} "
                  f"[{b['boot_pct'][0]:.2f},{b['boot_pct'][1]:.2f}] "
                  f"(n={b['k']}, correct={b['n_correct']:.0f}) vs 2xR = {2 * r:.3f} "
                  f"(R = {common.n_pct(int(round(r * blk['n'])), blk['n'])})")
    json.dump(verdicts, open(common.out_path("task1_verdicts.json"), "w"), indent=1)
    checks.append(common.check("verdict file written; SEND THESE NUMBERS NOW", True))
    if "K=200" in verdicts and verdicts["K=200"].startswith("GRACEFUL"):
        print("NOTE per pre-reg rule 3: a GRACEFUL K=200 contradicts the committed "
              "structure (oracle filter dead at K=200) -- bug-check the join before claiming.")

    common.finish(checks)


if __name__ == "__main__":
    main()
