"""
Metric 3 — predicate-flip survival / necessity.
=====================================================================
analysis/robustness.py (numpy only).  Two exact legs + one sampled leg:

    necessity (exact)     tree: flipping any single reason literal
                          breaks sufficiency (reachable-leaf DFS via
                          utils/tree_ops).  teacher: INSTANCE-level --
                          flipping the literal changes the teacher's
                          prediction for this image (the box-level
                          analogue is vacuous when reasons are not
                          teacher-sufficient; see metric 1).
    survival (sampled)    flip r random NON-reason predicates; fraction
                          of draws where tree and teacher still
                          predict the class.

Output: results/metric3_robustness_K{K}.csv
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

from utils.tree_ops import TreeOps  # noqa: E402

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
RESULTS_DIR = PROJECT_ROOT / "results"

SCHEMA = ["method", "K", "sample_id", "size",
          "tree_necessity_min1", "teacher_necessity_min1",
          "tree_survival_r1", "tree_survival_r2", "tree_survival_r4",
          "teacher_survival_r1", "teacher_survival_r2", "teacher_survival_r4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--tree", default=None)
    ap.add_argument("--reasons", default=None)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    teacher_path = args.teacher or str(
        Path(args.bank) / f"teacher_K{args.classes}.npz")
    tree_path = args.tree or str(Path(args.bank) / f"tree_K{args.classes}.json")
    reasons = args.reasons or str(
        RESULTS_DIR / f"reasons_K{args.classes}.jsonl")
    out = args.out or str(
        RESULTS_DIR / f"metric3_robustness_K{args.classes}.csv")

    teacher = np.load(teacher_path)
    W, b, classes = teacher["W"], teacher["b"], teacher["classes"]
    ops = TreeOps(json.loads(Path(tree_path).read_text()))
    te = np.load(Path(args.bank) / "bank_test.npz")

    rows = []
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
            K = rec["K"]
            mask = te["labels"] < K
            idx = int(np.where(te["sample_ids"][mask] == rec["sample_id"])[0][0])
            x = te["bits"][mask][idx].astype(np.int64)
            c = rec["pred_tree"]
            lits = [(abs(int(l)) - 1, 1 if int(l) > 0 else 0)
                    for l in raw]
            reason_feats = {f for f, _ in lits}

            tree_nec, teach_nec = [], []
            for f, bit in lits:
                flipped = [(ff, bb if ff != f else 1 - bb) for ff, bb in lits]
                tree_nec.append(not ops.sufficient(flipped, c))
                xv = x.copy().astype(np.float64)
                xv[f] = 1 - xv[f]
                logits = W @ xv + b
                teach_nec.append(
                    int(classes[int(np.argmax(logits))]) != c)

            rng = np.random.RandomState(rec["sample_id"] % (2 ** 31))
            outside = np.array([f for f in range(ops.n_features)
                                if f not in reason_feats])
            surv = {r: [0, 0] for r in (1, 2, 4)}  # [tree, teacher] hits
            for _ in range(args.draws):
                for r in (1, 2, 4):
                    if len(outside) < r:
                        continue
                    flips = rng.choice(outside, size=r, replace=False)
                    vec = x.copy()
                    for f in flips:
                        vec[f] = 1 - x[f]
                    if ops.predict_one(vec) == c:
                        surv[r][0] += 1
                    logits = W @ vec.astype(np.float64) + b
                    if classes[int(np.argmax(logits))] == c:
                        surv[r][1] += 1
            rows.append([
                "pyxai", K, rec["sample_id"], len(lits),
                int(all(tree_nec)), int(all(teach_nec)),
                round(surv[1][0] / args.draws, 4),
                round(surv[2][0] / args.draws, 4),
                round(surv[4][0] / args.draws, 4),
                round(surv[1][1] / args.draws, 4),
                round(surv[2][1] / args.draws, 4),
                round(surv[4][1] / args.draws, 4),
            ])
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMA)
        w.writerows(rows)
    summary = {
        "n": len(rows),
        "frac_tree_necessity_min1": round(
            float(np.mean([r[4] for r in rows])) if rows else None, 4),
        "frac_teacher_necessity_min1": round(
            float(np.mean([r[5] for r in rows])) if rows else None, 4),
        "mean_tree_survival_r1": round(
            float(np.mean([r[6] for r in rows])) if rows else None, 4),
        "mean_teacher_survival_r1": round(
            float(np.mean([r[9] for r in rows])) if rows else None, 4),
        "draws": args.draws,
    }
    Path(out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"--> metric3: tree necessity {summary['frac_tree_necessity_min1']}"
          f" | teacher necessity {summary['frac_teacher_necessity_min1']} | "
          f"survival r1 tree {summary['mean_tree_survival_r1']} "
          f"teacher {summary['mean_teacher_survival_r1']}", flush=True)


if __name__ == "__main__":
    main()
