"""
Metric 6 — maximum concept coverage at K (MHS set cover).
=====================================================================
analysis/max_coverage.py (numpy/stdlib).  The behavior-explainer
Metric-4 analogue on our banks: over the per-image sufficient reasons
(one set of concepts per test image), find the smallest set of
concepts that HITS every image's reason (a hitting set / set cover).
Solved three ways with utils/mhs.py (A exact, B random, C hybrid) and
reported side by side; also the greedy max-frequency cover as a
reference.  A small cover = a few concepts that touch every image's
explanation (a compact "global summary" of the model's reasons).

Output: results/metric6_max_coverage_K{K}{tag}.csv
        (solver, K, cover_size, cover, n_images_hit, mean_reason_size)
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.mhs import minimal_hitting_setA, minimal_hitting_setB, minimal_hitting_setC  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

SCHEMA = ["solver", "K", "cover_size", "cover",
          "n_images_hit", "mean_reason_size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasons", required=True)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(
        RESULTS_DIR / f"metric6_max_coverage_K{args.classes}.csv")

    sets, sizes = [], []
    with open(args.reasons) as fh:
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
            # hit on concept identity (sign-agnostic): a concept appears
            # in an image's reason with some polarity
            sets.append(sorted({abs(int(l)) for l in raw}))
            sizes.append(len(raw))
    if not sets:
        raise SystemExit("no reasons found")

    rows = []
    for solver, cover in (
        ("A_exact", minimal_hitting_setA(sets, [])),
        ("B_random", minimal_hitting_setB(sets, seed=0)),
        ("C_hybrid", minimal_hitting_setC(sets, [], attempts=100)),
    ):
        rows.append([solver, args.classes, len(cover),
                     ";".join(map(str, cover)), len(sets),
                     round(sum(sizes) / len(sizes), 3)])
    # reference: greedy max-frequency cover (counts recomputed over the
    # REMAINING sets each step -- a stale-count argmax can have zero
    # remaining coverage and would loop forever)
    greedy, remaining = [], [set(s) for s in sets]
    while remaining:
        counts = Counter(f for s in remaining for f in s)
        best = max(counts, key=lambda f: (counts[f], -f))
        greedy.append(best)
        remaining = [s for s in remaining if best not in s]
    rows.append(["greedy_freq", args.classes, len(greedy),
                 ";".join(map(str, greedy)), len(sets),
                 round(sum(sizes) / len(sizes), 3)])

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMA)
        w.writerows(rows)
    summary = {r[0]: r[2] for r in rows}
    Path(out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"--> metric6: cover sizes {summary} "
          f"over {len(sets)} reasons (mean size {rows[0][5]})", flush=True)


if __name__ == "__main__":
    main()
