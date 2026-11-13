# CVPR 2027 hand-off — the three runs (assignment brief)

**Issued:** 2026-11-11. Runner-facing assignment as written by the lead. Execution status: all three tasks have ready scripts at [`grounded_vision/scripts/tasks/`](../../grounded_vision/scripts/tasks/) (see that README for exact commands and the environment note); the scientific criteria were pre-registered BEFORE any run in [`CVPR2027_PREREG_TASK_RUNS.md`](CVPR2027_PREREG_TASK_RUNS.md) — read it first. Writing, abstract decisions, and the reconciliation of the two trust results stay with the lead; the runner delivers numbers.

---

## Task 1 (first, due in 1.5 days): calibrated abstention at more class counts

- Run the existing audit and uncertainty-aware reason pipeline at **K=20 and K=50**, and add the **calibrated arm at K=200**, which has never been measured.
- Then run `scripts/abstention_curves.py` on each.
- **Deliverable:** one table per K with coverage, joint premise support, number of instances and bootstrap CI for the four scores (**calibrated, raw, oracle marginal, random**), plus one figure of support against coverage with a curve per K.
- This decides whether the paper's claim is the tradeoff or the collapse, so it goes first.

Runner: `task1_calibrated_scaling.py` (pipelines → curves → figure → pre-registered GRACEFUL/CLIFF verdict lines).

## Task 2 (due day 3): sensitivity of the 4-class result

- Five random 4-class draws with fixed seeds, same pipeline.
- **Deliverable:** premise support per draw, and for each draw whether a single literal accounts for most failures and which one.
- This tells you whether "one literal carries every failure" is a property of your subset or a pattern.

Runner: `task2_subset_sensitivity.py` (draws fixed at seeds 1000–1004 via the new `--class-ids` pipeline option; dominance criterion pre-registered: literal in ≥80% of failing reasons AND marginal support ≤0.10; verdict PATTERN ≥3/5 draws).

## Task 3 (due day 4): the forcing sweep

- For each threshold, check whether the trusted literals **force** the prediction, **minimise to a reason**, and score support on both the minimal reason and the whole trusted set.
- **Deliverable:** threshold, coverage and both support numbers, **compared against the ranking rule at the same threshold**.
- This both answers the exact-query question and should explain why `guarantee_scaling` reports 0.07–0.10.

Runner: `task3_forcing_sweep.py` (δ ∈ {0.5…0.9}, K ∈ {4,10}; greedy-irreducible minimisation, documented; prints the side-by-side with the committed `guarantee_scaling` trusted-only numbers; interpretation reserved to the lead).

## Ground rules to hand over with the tasks

1. **Write new result files and never overwrite committed ones.** (Enforced in code: output paths refuse to overwrite.)
2. **Dump per-instance records, not only aggregates.**
3. **Fit calibrators on validation only.**
4. **Report the number of instances beside every percentage.**
5. **Send Task 1 numbers as soon as they exist, even if they look bad.**

## If finished early

The next job is the comparison table against the **reasoning-shortcuts (Marconato et al.)** and **Teso et al.** papers, since citations are due on the 15th. Ask the lead for the brief first — the related-work notes start from zero (nothing is in the repo).

## The cliff decision (made before the runs, per the lead's rule)

Recorded in [`CVPR2027_PREREG_TASK_RUNS.md`](CVPR2027_PREREG_TASK_RUNS.md) before Task 1 executes: **CLIFF at K** = upper 95% bootstrap bound of calibrated-arm support at 14% coverage falls below **2× the unconditional rate R_K**; **GRACEFUL** = lower bound ≥ 2×R_K; ambiguous CIs decided by the curve average over 5–30% coverage. Abstract-change rules committed with it (including: a GRACEFUL K=200 triggers a bug-check before celebration, since the oracle filter is dead at K=200).
