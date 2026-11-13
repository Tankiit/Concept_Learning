# Hand-off tasks, Nov 11–16 (CVPR cut)

Three runs, in priority order. Everything runs under the repo venv and reuses
the committed pipeline scripts; pass/fail lines print as `PASS:` / `FAIL:` and
the scripts exit non-zero on mechanical failure. The scientific criteria are
pre-registered in `docs/wiki/CVPR2027_PREREG_TASK_RUNS.md` (read it first —
the thresholds were fixed before you run anything, on purpose).

```bash
cd grounded_vision
V=/Users/tanmoy/anaconda3/bin/python   # full stack: torch, torchvision, sklearn, mpl

# Task 1 — FIRST, due in 1.5 days (K=20/50 audit + X2, K=200 calibrated arm)
$V scripts/tasks/task1_calibrated_scaling.py

# Task 2 — due day 3 (five random 4-class draws)
$V scripts/tasks/task2_subset_sensitivity.py

# Task 3 — due day 4 (forcing sweep; explains the guarantee_scaling 0.07–0.10)
$V scripts/tasks/task3_forcing_sweep.py
```

All three default to the local cache/data paths; override with
`--cache-dir` / `--data-dir` if running elsewhere. Runtime note: Task 1's X2
leg at K=200 processes the full 5,794-image test set with Monte-Carlo
completions — if it is too slow, rerun with `--x2-extra "--B 200"` (the
calibrated min-confidence score does not depend on B) and say so when sending
numbers.

## Ground rules (enforced in code)

1. **New files only.** Every output goes to a fresh `results/task*` file; the
   scripts refuse to overwrite anything that already exists.
2. **Per-instance records, not just aggregates.** Every task dumps per-instance
   JSON; the tables are derived from those dumps.
3. **Calibrators fit on validation only.** The val carve-out (15%) is the only
   thing that touches isotonic fitting; test instances are never used for it.
4. **Every percentage printed beside its n** (`k/n = 0.xxx`).
5. **Task 1 numbers get sent the moment they exist**, even if they look bad —
   especially if they look bad.

## What comes back

- Task 1: one table per K (coverage, joint premise support, n, bootstrap CI
  for calibrated / raw / oracle-marginal / random), the verdict line per K
  (GRACEFUL / CLIFF / AMBIGUOUS per the pre-registration), and
  `results/figures/task1_abstention_scaling.{png,pdf}`.
- Task 2: P(all) per draw with Wilson CI, the dominating literal per draw and
  its share/marginal, and the PATTERN / SUBSET-SPECIFIC verdict.
- Task 3: the δ-threshold table per K (coverage and both support numbers plus
  the ranking rule) and the side-by-side with the committed
  `guarantee_scaling` trusted-only numbers. Interpretation of that table
  stays with the lead — send the numbers, not a conclusion.

## If you finish early

The next job is the comparison table against reasoning-shortcuts (Marconato
et al.) and Teso et al. — citations are due Nov 15, and the related-work notes
start from zero (nothing is in the repo). Ask for the brief first.
