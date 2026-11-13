#!/usr/bin/env python3
"""Task 2 (due day 3): sensitivity of the four-class premise-support result.

Five random 4-class draws (seeds 1000-1004, ids sampled from 0..199, recorded in
the output), audit pipeline at K=4 setting T per draw, then per draw:
  - P(all premises supported) with Wilson CI and n
  - whether ONE literal dominates the failures (>=80% of failing reasons AND
    marginal support <= 0.10 -- pre-registered) and which one
Aggregate verdict per pre-registration: >=3/5 draws dominated => pattern.

Never overwrites: each draw writes task2_draw{i}.json; analysis writes
task2_subset_sensitivity.json. Percentages always printed with their n.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402

DRAW_SEEDS = [1000, 1001, 1002, 1003, 1004]


def analyse_block(block):
    pi = [r for r in block["per_instance"] if r["literals"]]
    n = len(pi)
    n_sup = sum(1 for r in pi if r["all_supported"])
    lo, hi = common.wilson(n_sup, n)
    failing = [r for r in pi if not r["all_supported"]]
    occ = defaultdict(list)
    for r in failing:
        for l in r["literals"]:
            occ[(l["feature"], l["bit"])].append(bool(l["supported"]))
    top = None
    if failing:
        # literal carried by the largest share of failing reasons
        counts = defaultdict(int)
        for r in failing:
            for key in {(l["feature"], l["bit"]) for l in r["literals"]}:
                counts[key] += 1
        key = max(counts, key=lambda k: counts[k])
        marginal = float(np.mean(occ[key])) if occ[key] else None
        top = {"literal": f"{key[0]}|{key[1]}", "share_of_failing": counts[key] / len(failing),
               "marginal_support": marginal,
               "n_failing": len(failing), "n_failing_with_it": counts[key]}
        dominates = top["share_of_failing"] >= 0.8 and marginal is not None and marginal <= 0.10
    else:
        dominates = False
    return {"n": n, "n_all_supported": n_sup, "p_all": n_sup / n if n else None,
            "wilson95": [lo, hi], "top_literal": top, "dominates": dominates}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=common.DEFAULT_CACHE)
    ap.add_argument("--data-dir", default=common.DEFAULT_DATA)
    args = ap.parse_args()
    checks = []

    summary = []
    for i, seed in enumerate(DRAW_SEEDS):
        ids = sorted(int(x) for x in np.random.RandomState(seed).choice(200, 4, replace=False))
        ids_s = ",".join(str(x) for x in ids)
        out = common.out_path(f"task2_draw{i}_seed{seed}_ids{ids_s.replace(',', '-')}.json")
        common.run_pipeline("run_cbm_cem_audit.py", out, args.cache_dir, args.data_dir,
                            extra=["--classes", "4", "--class-ids", ids_s,
                                   "--settings", "T", "--seeds", "42"])
        d = json.load(open(out))
        blk = next(b for b in d["blocks"] if b["setting"] == "T_tree_head")
        res = analyse_block(blk)
        res.update({"draw": i, "seed": seed, "class_ids": ids, "file": os.path.basename(out)})
        summary.append(res)
        t = res["top_literal"]
        print(f"draw {i} (seed {seed}, ids {ids_s}): "
              f"P(all) = {common.n_pct(res['n_all_supported'], res['n'])} "
              f"[{res['wilson95'][0]:.3f}, {res['wilson95'][1]:.3f}]"
              + (f" | top literal {t['literal']} in {t['n_failing_with_it']}/{t['n_failing']}"
                 f" failing (share {t['share_of_failing']:.2f}, marginal {t['marginal_support']:.3f})"
                 f" -> {'DOMINATES' if res['dominates'] else 'no'}" if t else " | no failures"))
        checks.append(common.check(f"draw {i}: per-instance dump present (n={res['n']})",
                                   res["n"] > 0))

    n_dom = sum(1 for r in summary if r["dominates"])
    verdict = "PATTERN" if n_dom >= 3 else "SUBSET-SPECIFIC"
    print(f"\n== Pre-registered verdict: {verdict} ({n_dom}/5 draws dominated) ==")
    committed = 0.2542
    for r in summary:
        if r["wilson95"][0] > committed or r["wilson95"][1] < committed:
            print(f"  note: draw {r['draw']} Wilson CI excludes the committed 25.4% "
                  f"[{r['wilson95'][0]:.3f}, {r['wilson95'][1]:.3f}] -- body-only flag")

    out = common.out_path("task2_subset_sensitivity.json")
    json.dump({"draw_seeds": DRAW_SEEDS, "prereg": ">=3/5 dominated => pattern",
               "verdict": verdict, "draws": summary}, open(out, "w"), indent=1)
    checks.append(common.check(f"summary written to {os.path.basename(out)}", True))
    common.finish(checks)


if __name__ == "__main__":
    main()
