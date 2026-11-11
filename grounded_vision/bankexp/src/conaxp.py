"""
conaxp.py — erasure-based baseline, reimplemented on our bank.
================================================================
Stage 5b (numpy only).  Classic greedy backward erasure for the LINEAR
teacher: start with ALL bank features fixed to the instance's bits;
repeatedly erase (unfix) one feature while the remaining fixed set
still EXACTLY forces the teacher's class (box-LP oracle from
utils/box_lp.py).  Two deletion orders:

    random        canonical backward-erasure baseline (seeded)
    contribution  drop the feature with the smallest |w_c| first

The baseline is for the TEACHER (greedy, order-dependent); tree
reasons are already exact, so erasure is only run on the teacher and
compared on size + exact sufficiency (results/metric4_*).

Usage
-----
    python src/conaxp.py --classes 4 --order random
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

SCHEMA = ["method", "K", "sample_id", "size", "sufficient_exact", "order"]


def greedy_erasure(W, b, classes, x, target, order, rng):
    """Backward erasure: repeatedly drop a fixed feature while the
    remaining fixed set still exactly forces the teacher's class.
    Restarts the candidate order after every successful erase."""
    fixed = list(range(len(x)))
    c_idx = int(np.where(classes == target)[0][0])
    changed = True
    while changed and fixed:
        changed = False
        if order == "random":
            candidates = list(fixed)
            rng.shuffle(candidates)
        else:
            candidates = sorted(fixed, key=lambda f: abs(W[c_idx][f]))
        for f in candidates:
            if f not in fixed:
                continue
            trial = [g for g in fixed if g != f]
            lits = [(g, int(x[g])) for g in trial]
            ok, _ = reason_sufficient_exact(W, b, classes, lits, target)
            if ok:
                fixed = trial
                changed = True
                break
    return [(f, int(x[f])) for f in fixed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--order", choices=["random", "contribution"],
                    default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    teacher_path = args.teacher or str(
        Path(args.bank) / f"teacher_K{args.classes}.npz")
    out = args.out or str(
        RESULTS_DIR / f"metric4_erasure_conaxp_K{args.classes}_{args.order}.csv")

    teacher = np.load(teacher_path)
    W, b, classes = teacher["W"], teacher["b"], teacher["classes"]
    te = np.load(Path(args.bank) / "bank_test.npz")
    mask = te["labels"] < args.classes
    ids = te["sample_ids"][mask]
    X = te["bits"][mask].astype(np.int64)
    if args.limit:
        ids, X = ids[: args.limit], X[: args.limit]

    rng = np.random.RandomState(args.seed)
    rows = []
    for i in range(len(ids)):
        x = X[i]
        logits = W @ x.astype(np.float64) + b
        pred = int(classes[int(np.argmax(logits))])
        lits = greedy_erasure(W, b, classes, x, pred, args.order, rng)
        ok, _ = reason_sufficient_exact(W, b, classes, lits, pred)
        rows.append(["conaxp", args.classes, int(ids[i]), len(lits),
                     int(ok), args.order])
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMA)
        w.writerows(rows)
    summary = {
        "method": f"conaxp-{args.order}",
        "n": len(rows),
        "mean_size": round(float(np.mean([r[3] for r in rows])), 4) if rows else None,
        "frac_sufficient_exact": round(
            float(np.mean([r[4] for r in rows])) if rows else None, 4),
    }
    Path(out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"--> conaxp ({args.order}): mean size {summary['mean_size']} | "
          f"exact-sufficient {summary['frac_sufficient_exact']}", flush=True)


if __name__ == "__main__":
    main()
