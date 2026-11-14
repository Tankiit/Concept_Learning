#!/usr/bin/env python3
"""Ablation D1: gate aggregator -- min vs product vs mean of per-literal trust.

Pre-registered as P-A1 in CVPR2027_PREREG_TASK_RUNS.md (2026-11-14, before
this run): min ~= product at K in {4,10} (Spearman >= 0.9, CIs overlap);
mean worse or equal. Source: committed task3_forcing_sweep.json per-instance
literals (calibrated polarity-aware p_true + annotation support).
K=200 is not covered by that dump -- stated in the output.

Output: results/ablation_aggregator.json (refuses to overwrite).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "ablation_aggregator.json")
COVERAGES = [0.05, 0.10, 0.14, 0.20, 0.30]
AGGS = ("min", "product", "mean")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING to overwrite {OUT}")
    d = json.load(open(os.path.join(ROOT, "results", "task3_forcing_sweep.json")))
    rng = np.random.default_rng(42)
    out_blocks = []
    for blk in d["blocks"]:
        rows = [r for r in blk["per_instance"] if r["literals"]]
        y = np.array([all(l["supported"] for l in r["literals"]) for r in rows], float)
        scores = {a: np.array([
            {"min": min, "product": np.prod, "mean": np.mean}[a](
                [l["p_true"] for l in r["literals"]]) for r in rows], float)
            for a in AGGS}
        n = len(rows)
        R = float(y.mean())

        def at_cov(yy, ss, c):
            k = max(1, int(round(c * len(yy))))
            return float(yy[np.argsort(-ss)[:k]].mean())

        boot = {a: [] for a in AGGS}
        for _ in range(10000):
            idx = rng.integers(0, n, n)
            for a in AGGS:
                ratio = at_cov(y[idx], scores[a][idx], 0.14) / max(1e-9, y[idx].mean())
                boot[a].append(ratio)
        cell = {"K": blk["K"], "n": n, "unconditional_R": R,
                "sp_min_product": spearman(scores["min"], scores["product"]),
                "at_coverage": {}}
        for c in COVERAGES:
            cell["at_coverage"][f"{c:.2f}"] = {
                a: {"support": round(at_cov(y, scores[a], c), 4)} for a in AGGS}
        cell["LB_ratio_at_14pct"] = {
            a: round(float(np.percentile(boot[a], 2.5)), 3) for a in AGGS}
        cell["LB_ratio_2x_bar"] = round(2 * R, 4)
        out_blocks.append(cell)
        print(f"K={blk['K']} n={n} R={R:.4f} spearman(min,product)="
              f"{cell['sp_min_product']:.3f}")
        for c in COVERAGES:
            s = cell["at_coverage"][f"{c:.2f}"]
            print(f"  cov {c:.2f}: " + "  ".join(f"{a}={s[a]['support']:.3f}" for a in AGGS))
        print("  LB ratio @14%:", cell["LB_ratio_at_14pct"], "| 2x bar:", cell["LB_ratio_2x_bar"])
    json.dump({"prereg": "P-A1 (CVPR2027_PREREG_TASK_RUNS.md, fixed before run)",
               "note": "task3 dump covers K in {4,10} only; K=200 aggregator unmeasured",
               "blocks": out_blocks}, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
