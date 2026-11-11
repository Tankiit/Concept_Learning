"""
Metric 1 — exact sufficiency of tree reasons for the linear teacher.
=====================================================================
analysis/margin_sufficiency.py (numpy only; no model loading — reads
the teacher weight matrix and the precomputed PyXAI reasons).

A reason R is EXACTLY sufficient for class c under the linear teacher
iff for every competing class d, the box-max of
margin_d(x) = (w_d - w_c) . x + (b_d - b_c) is negative (utils/box_lp).
This is the exact counterpart of metric 2 (measured agreement).

Output: results/metric1_margin_sufficiency_K{K}.csv
        (method, K, sample_id, size, sufficient_exact, worst_margin,
         n_classes_checked)
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.box_lp import reason_sufficient_exact  # noqa: E402

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
RESULTS_DIR = PROJECT_ROOT / "results"

SCHEMA = ["method", "K", "sample_id", "size",
          "sufficient_exact", "worst_margin", "n_classes_checked"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--reasons", default=None)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eps", type=float, default=0.0)
    args = ap.parse_args()
    teacher_path = args.teacher or str(
        Path(args.bank) / f"teacher_K{args.classes}.npz")
    reasons = args.reasons or str(
        RESULTS_DIR / f"reasons_K{args.classes}.jsonl")
    out = args.out or str(
        RESULTS_DIR / f"metric1_margin_sufficiency_K{args.classes}.csv")

    teacher = np.load(teacher_path)
    W, b, classes = teacher["W"], teacher["b"], teacher["classes"]
    rows, n_suff = [], 0
    with open(reasons) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            raw = rec.get("pyxai_reason")
            method = "pyxai"
            if raw is None and rec.get("native_reason"):
                raw = [(f + 1) if b else -(f + 1)
                       for f, b in rec["native_reason"]]
                method = "native"
            if not raw:
                continue
            lits = [(abs(int(l)) - 1, 1 if int(l) > 0 else 0)
                    for l in raw]
            ok, worst = reason_sufficient_exact(
                W, b, classes, lits, rec["pred_tree"], eps=args.eps)
            n_suff += ok
            rows.append([method, rec["K"], rec["sample_id"],
                         len(lits), int(ok), round(worst, 4), len(classes) - 1])
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMA)
        w.writerows(rows)
    summary = {
        "method": "pyxai", "n_reasons": len(rows),
        "frac_sufficient_exact_teacher": round(n_suff / max(1, len(rows)), 4),
        "eps": args.eps,
        "note": ("exact box-LP: free coordinates set at the vertex "
                 "maximizing each competing class's linear margin"),
    }
    Path(out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"--> metric1: {n_suff}/{len(rows)} PyXAI tree-reasons exactly "
          f"sufficient for the linear teacher "
          f"({summary['frac_sufficient_exact_teacher']})", flush=True)


if __name__ == "__main__":
    main()
