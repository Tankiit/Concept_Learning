"""
Metric 2 — tree-vs-teacher agreement inside a reason.
=====================================================================
analysis/coverage.py (numpy only; reads precomputed reasons + teacher
weights).  The MEASURED transfer leg: for each reason R, over test
images whose bank bits satisfy R, what fraction does the frozen linear
teacher also predict the tree's class?  Exact tree sufficiency says
nothing about the teacher; this is the finite-sample bridge (Wilson
95% intervals, minimum group size).

Output: results/metric2_coverage_K{K}.csv
        (method, K, reason, size, class, n_satisfying,
         teacher_agreement, wilson_lo, wilson_hi)
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
RESULTS_DIR = PROJECT_ROOT / "results"

SCHEMA = ["method", "K", "reason", "size", "class", "n_satisfying",
          "teacher_agreement", "wilson_lo", "wilson_hi"]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--reasons", default=None)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--min-group", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    teacher_path = args.teacher or str(
        Path(args.bank) / f"teacher_K{args.classes}.npz")
    reasons = args.reasons or str(
        RESULTS_DIR / f"reasons_K{args.classes}.jsonl")
    out = args.out or str(RESULTS_DIR / f"metric2_coverage_K{args.classes}.csv")

    teacher = np.load(teacher_path)
    W, b, classes = teacher["W"], teacher["b"], teacher["classes"]
    te = np.load(Path(args.bank) / "bank_test.npz")

    seen = OrderedDict()
    with open(reasons) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            raw = rec.get("pyxai_reason")
            if raw is None and rec.get("native_reason"):
                raw = [(f + 1) if b else -(f + 1)
                       for f, b in rec["native_reason"]]
            if not raw:
                continue
            seen.setdefault(tuple(sorted(raw)), rec)
    rows = []
    for key, rec in seen.items():
        K = rec["K"]
        mask = te["labels"] < K
        X = te["bits"][mask].astype(np.float64)
        lits = [(abs(int(l)) - 1, 1 if int(l) > 0 else 0) for l in key]
        sat = np.ones(len(X), dtype=bool)
        for f, bit in lits:
            sat &= X[:, f] == bit
        n = int(sat.sum())
        if n < args.min_group:
            continue
        logits = X[sat] @ W.T + b
        tpred = classes[np.argmax(logits, axis=1)]
        hits = int((tpred == rec["pred_tree"]).sum())
        lo, hi = wilson(hits, n)
        rows.append(["pyxai", K, ";".join(f"{l:+d}" for l in key),
                     len(key), rec["pred_tree"], n, round(hits / n, 4), lo, hi])
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMA)
        w.writerows(rows)
    summary = {
        "n_reason_patterns": len(rows), "min_group": args.min_group,
        "mean_teacher_agreement": round(
            float(np.mean([r[6] for r in rows])) if rows else None, 4),
        "median_group_n": int(np.median([r[5] for r in rows])) if rows else None,
    }
    Path(out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"--> metric2: {len(rows)} reason patterns (min group "
          f"{args.min_group}); mean in-reason teacher agreement "
          f"{summary['mean_teacher_agreement']}", flush=True)


if __name__ == "__main__":
    main()
