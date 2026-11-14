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

---

## EXECUTION RESULTS (2026-11-14, run by the lead locally; all three tasks mechanical PASS)

Environment: conda `torch-multimodal` (torch 2.10 / torchvision 0.25 / sklearn 1.5.1) on the local flip-free eval-transform caches, seeds and paths exactly as pre-registered. Two committed-script bugs surfaced and were fixed during the runs (both fixes preserve default behavior bit-for-bit; details in the hand-off page): `run_x2_uncertainty.py` crashed on any `--classes` without K=4 (verdict block hardcoded `blk("T_tree_head", 4)` after all compute, before the dump), and `run_cbm_cem_audit.py --class-ids` never remapped labels to 0..K−1 (crash for any draw containing ids ≥ K; silent wrong-problem training for all-small draws).

### Task 1 — verdicts (per K, B@0.14 vs 2×R, cluster-bootstrap 95% CI)

| K | R (unconditional) | B@0.14 [CI], n | 2×R | Verdict |
| --- | --- | --- | --- | --- |
| 20 | 36/515 = 0.070 | 0.361 [0.24, 0.47], n=72 (26 correct) | 0.140 | **GRACEFUL** |
| 50 | 21/600 = 0.035 | 0.143 [0.07, 0.23], n=84 (12 correct) | 0.070 | **GRACEFUL** |
| 200 | 10/600 = 0.017 | 0.095 [0.04, 0.15], n=84 (8 correct) | 0.033 | GRACEFUL **→ rule 3** |

- **Rule 1 fires** (K=20 AND K=50 both GRACEFUL): abstract unchanged; body gains the scaling curve (figure `results/figures/task1_abstention_scaling.{png,pdf}`); the tradeoff extends to intermediate scale.
- **Rule 3 fires at K=200** (the 0.14-coverage lower bound 0.04 ≥ 2×R=0.033; the 0.05 and 0.10 cells do NOT clear it). The prescribed bug-check was executed and is CLEAN: sid-join exact (600 rows, 0 reason mismatches), **zero structural-only reasons** at K=200 (B cannot be inflated by `has_part` literals), and the committed-structure claim re-verified (95% of the 56 reason-literal marginals strictly between 0 and 1 — the *oracle* marginal filter is dead; the *calibrated* per-instance filter is a different mechanism and still separates, B ≈ 5.6×R at 0.14 coverage). Per rule 3 the abstract is NOT strengthened; the K=200 cell is coverage-fragile (only one coverage clears the threshold) and the decision stays with the lead.
- Full curve tables (A_ir / A_cq / B / C / D at 6 coverages, per K): `results/task1_abstention_K20_50.json`, `results/task1_abstention_K200.json`; sources `task1_cbcem_K20_50.json`, `task1_x2_K20_50_200.json`. Note: the audit's D (distilled) arm is T-only beyond K=20 by design (`K <= 20` guard in the audit script), so K=50/200 curves are Setting T only.

### Task 2 — verdict: **SUBSET-SPECIFIC** (0/5 draws dominated)

| Draw | seed | class ids | P(all) [Wilson 95] | top literal in failing | share | marginal | dominates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1000 | 70,118,161,163 | 40/119 = 0.336 [0.258, 0.425] | `beak:bill_color=yellow\|0` | 0.62 | 0.918 | no |
| 1 | 1001 | 100,116,127,133 | 47/109 = 0.431 [0.342, 0.525] | `crown:crown_color=white\|0` | 0.94 | 0.897 | no |
| 2 | 1002 | 14,49,76,87 | 32/118 = 0.271 [0.199, 0.358] | `beak:bill_color=yellow\|0` | 0.65 | 0.946 | no |
| 3 | 1003 | 11,19,30,59 | 44/114 = 0.386 [0.302, 0.478] | `beak:bill_color=yellow\|0` | 0.83 | 0.983 | no |
| 4 | 1004 | 101,107,108,184 | 32/120 = 0.267 [0.196, 0.352] | `crown:crown_color=white\|0` | 0.68 | 0.933 | no |

- Per the pre-registered ≤2/5 branch: the abstract sentence ("a single literal carries all of them in our smallest setting") **stays** — it is true of the audited subset — and the **body must report the subset-dependence explicitly**; no generalisation of the one-literal claim.
- No draw exceeds 50% (max 0.431), so the abstract's "rarely all true" is robust to the spread. Draws 0, 1, 3 have Wilson CIs excluding the committed 25.4% — flagged body-only, as pre-registered.
- Observation for the lead (numbers, not a conclusion): the failing reasons in every draw DO concentrate on one literal (shares 0.62–0.94), but the carriers are **negated** clauses (`...=yellow|0`, `...=white|0`) with **high** marginal support (0.90–0.98) — the mirror image of the committed subset's positive `bill_color=yellow` at 0/88. The concentration phenomenon recurs; the pre-registered (≥0.80 share AND ≤0.10 marginal) profile does not.

### Task 3 — the forcing sweep (numbers; interpretation stays with the lead)

Setting T, calibrated polarity-aware p_true (val-fit only), greedy-irreducible minimisation, per-instance dumps (`results/task3_forcing_sweep.json`):

| K | δ | cov_forcing | support_kept\|forcing | support_minimal | cov_ranking | support\|ranking |
| --- | --- | --- | --- | --- | --- | --- |
| 4 (n=118) | 0.5–0.9 (all) | 30/118 = 0.254 | 30/30 = 1.000 | 1.000 | 0.254 | 1.000 |
| 10 (n=243) | 0.5 | 32/243 = 0.132 | 23/32 = 0.719 | 0.719 | 0.132 | 0.719 |
| 10 | 0.6 | 22/243 = 0.091 | 15/22 = 0.682 | 0.682 | 0.091 | 0.682 |
| 10 | 0.7 | 14/243 = 0.058 | 8/14 = 0.571 | 0.571 | 0.058 | 0.571 |
| 10 | 0.8 | 12/243 = 0.049 | 8/12 = 0.667 | 0.667 | 0.049 | 0.667 |
| 10 | 0.9 | 8/243 = 0.033 | 6/8 = 0.750 | 0.750 | 0.033 | 0.750 |

Against the committed `guarantee_scaling` trusted-only at K=4, δ=0.7–0.9: same coverage (0.254, n=30) but P(kept true|covered) = 0.067–0.133 there vs **1.000 here**. Pre-registered expectation outcomes: **E1 rejected** (cov_forcing ≡ cov_ranking at every δ, both K — the forcing event adds nothing beyond the ranking event); **E2 rejected** (support-on-kept is 1.000 at K=4, 0.57–0.75 at K=10 — nowhere near 0.07–0.10); **E3 exact-equality** (support_minimal ≡ support_kept in every cell — reasons are already near-irreducible). The pre-registered alternative branch therefore fires: **the 0.07–0.10 in `guarantee_scaling` is the selection/coverage event (raw-confidence trusted literals), not support given the calibrated selection.** The K=4 curve is a step function (identical at every δ): the only coverable instances are the 30 all-correct pipit reasons.
