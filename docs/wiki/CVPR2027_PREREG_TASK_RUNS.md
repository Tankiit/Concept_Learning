# Pre-registration: the three hand-off runs (Tasks 1–3)

**Written 2026-11-11, BEFORE any of the three runs execute.** These criteria are fixed before the numbers exist; the abstract-change rules are commitments, not options. Companion: [`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) keep list, [`CVPR2027_SUBMISSION_PLAN.md`](CVPR2027_SUBMISSION_PLAN.md) §9. Scripts live in `grounded_vision/scripts/tasks/` (student hand-off; ground rules in that README).

---

## Task 1 — calibrated abstention at K = 20, 50, 200

**Metric:** calibrated-arm joint premise support (B) at coverage **c = 0.14** with cluster-bootstrap 95% CI (10k resamples, instances resampled), against **R_K = unconditional P(all premises supported)** measured in the same run. Every percentage reported with its n.

**Verdicts (per K):**

- **GRACEFUL** at K: lower CI bound of B@0.14 ≥ **2 × R_K**.
- **CLIFF** at K: upper CI bound of B@0.14 < **2 × R_K**.
- **AMBIGUOUS** (CI straddles 2×R_K): decide by B averaged over c ∈ {0.05, 0.10, 0.14, 0.20, 0.30}, same 2× rule on the average.

**Why 2×:** at K=10 the calibrated arm sits ≈ 5.6 × R (0.697 vs 0.124). If the margin halves per scale step (5.6 → 2.8 → 1.4) it crosses 2× between K=20 and K=50; 2× is the smallest margin separable at n ≈ 240–600 and the smallest a practitioner would call a repair rather than noise.

**Abstract-change rules (committed now):**

1. K=20 AND K=50 both GRACEFUL → abstract unchanged; body gains the scaling curve; the tradeoff claim extends to intermediate scale.
2. Any of K=20/50 CLIFF → contribution (4) of the abstract gains "and the tradeoff collapses as classes grow"; the paper's spine shifts to the collapse and the scaling curve becomes the figure. K=200 is run regardless to complete the curve.
3. B at K=200 with any coverage ≥ 5% whose lower bound ≥ 2 × R_200 → **do not strengthen the abstract yet**: this contradicts the committed structure (the oracle reliability filter is dead at K=200, marginals diffuse — 95% of literals strictly between 0 and 1). First re-check literal marginal supports and the sid-join, then decide.
4. AMBIGUOUS at both 20 and 50 → keep the abstract as-is, report the curve, and spend no further compute chasing it before Nov 16.

**K=200 join note:** the committed audit at K=200 has per-instance records for 600 of 5,794 test images; the new X2 run covers the full test set. The join keeps the audited 600; report n after the join. If the X2 K=200 run is prohibitively slow, subsample deterministically (seed 42, first 600 by sid order) and report n.

## Task 2 — sensitivity of the four-class result

**Protocol:** five draws, seeds 1000–1004, four distinct class ids sampled uniformly from 0..199 per draw (recorded in the output), pipeline seed 42, setting T, linear reader.

**"One literal dominates a draw":** some literal appears in ≥ 80% of that draw's failing reasons AND has marginal annotation support ≤ 0.10.

**Verdicts:**

- ≥ 3/5 draws dominated → **pattern**: the body may say "typically one or a few positive literals dominate"; the abstract's existing sentence stands.
- ≤ 2/5 draws dominated → **subset-specific**: the abstract sentence ("a single literal carries all of them in our smallest setting") stays — it is true of the audited subset — but the body must report the subset-dependence explicitly, and no generalisation of the one-literal claim is made.
- P(all) per draw reported with Wilson CI; a draw whose CI excludes the committed 25.4% is flagged in the body only (the abstract's "rarely all true" is robust to the spread; verify by inspection that no draw exceeds 50%).

## Task 3 — the forcing sweep (K = 4, 10; δ ∈ {0.5, 0.6, 0.7, 0.8, 0.9})

**Deliverable-complete =** per K and δ: coverage_forcing (trusted subset forces the prediction), support on the kept set, support on the minimal trusted reason (greedy irreducibility over kept literals — documented as greedy, not exact), plus the ranking rule at the same δ (all reason literals ≥ δ) with its coverage and whole-reason support. Per-instance dump written. All counts beside all percentages.

**Pre-registered expectations (interpretation stays with the lead, not the runner):**

- E1: coverage_forcing ≪ coverage_ranking at every δ (forcing is the strictly harder event — it requires sufficiency, not just trust).
- E2: support-on-kept is low because the kept subsets that force are dominated by high-p_true literals that are nonetheless annotation-false (per-concept calibration cannot see polarity errors like a reader confidently asserting a bill color that the annotation rejects). This is the candidate explanation of `guarantee_scaling`'s 0.07–0.10.
- E3: support-on-minimal ≥ support-on-kept (minimisation drops literals, and dropped literals are disproportionately the supported-but-unneeded ones).
- If support-on-kept ≈ support-ranking at matched coverage instead → the 0.07–0.10 discrepancy is the coverage event, not support. If neither E-hypothesis fits → unexplained; that is a finding about the discrepancy, not a failure of the task.

**Mechanical PASS for all three tasks** = scripts exit 0, new result files written (no existing file touched), per-instance dumps present, every percentage accompanied by n. Scientific verdicts print from these pre-registered thresholds, but exit codes encode only mechanical success.
