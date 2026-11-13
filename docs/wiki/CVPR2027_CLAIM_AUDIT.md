# CVPR 2027 claim audit — eight questions against the "invariant claim" abstract

**Drafted:** 2026-11-11. An external review of the proposed cross-dataset invariant claim ("certificate exact everywhere, premises fail everywhere, invariant to vocabulary/reader/K/selection") raised eight questions. This page records the verdict per question, the new numbers computed from committed per-instance data, and what survives for the paper. Companion to [`CVPR2027_SUBMISSION_PLAN.md`](CVPR2027_SUBMISSION_PLAN.md) (§9 abstract updated the same day).

**New artifact:** [`independence_gap_cub.json`](../../grounded_vision/results/independence_gap_cub.json) via [`scripts/independence_gap_cub.py`](../../grounded_vision/scripts/independence_gap_cub.py) — reads the committed `per_instance` dumps in `cbm_cem_audit.json` (K=4/10, settings T and D, n=118 each) and `cbm_cem_audit_K200.json` (n=600 audited of 5,794). No models were re-run; everything below is analysis of committed records.

---

## The headline correction

The proposed abstract sentence — *verified exact in 100% of audited instances on every dataset* — is **true by construction** wherever it was measured: an abductive explanation (AXp) of a tree is sufficient by definition, and the native reachable-leaf DFS check is the same arithmetic. It is a passing unit test, not a finding. Where exactness was genuinely at risk and was defended: the PyXAI 2.0.1 engine (four documented bugs, caught by the independent native verifier; PyXAI-vs-native reason-set match only 0.84–0.97 — minimality choice differs, sufficiency never failed), and the CEM embedding head, which the audit protocol deliberately excludes from certification. The defensible paper quantities are the **transfer rungs** (fidelity, premise support), not the certificate-exactness count.

## Q1 — what would `verified_frac < 1.0` have meant?

Conceded: a unit test. Evidence kept as sanity material: `verified_cert`/`verified_dfs` agreement, the 472/472 real-PyXAI cross-check, the engine-bug ledger. Paper phrasing: certificates are exact *where they apply by construction*; the measurements are the transfer rungs.

## Q2 — is 25% → 7% → 1.7% just compounding? (computed 2026-11-11)

**No — observed joint support sits ABOVE the independence prediction, and at K=4 the comparison is degenerate because the failure is one literal.**

| block | n | mean len | observed | independence-predicted | ratio obs/pred [95% CI] | cross-fit ratio |
| --- | --- | --- | --- | --- | --- | --- |
| T / K=4 | 118 | 3.61 | 0.254 | 0.254 | 1.000 (degenerate) | 1.00 |
| D / K=4 | 118 | 2.57 | 0.254 | 0.254 | 1.000 (degenerate) | 1.00 |
| T / K=10 | 243 | 5.88 | 0.115 | 0.076 | **1.52 [1.23, 1.89]** | 1.48 |
| D / K=10 | 243 | 3.69 | 0.124 | 0.114 | 1.09 [0.99, 1.21] | 1.19 |
| T / K=200 | 600 | 10.76 | 0.017 | 0.014 | 1.24 [0.56, 2.00] n.s. | 1.20 |

Bootstrap: cluster over instances, 10k resamples, marginals re-estimated per resample. Cross-fit: 2-fold, marginals estimated out-of-half.

**K=4 structural finding:** all 88 failing instances contain `beak:bill_color=yellow|1` (marginal annotation support exactly 0/88); all 30 passing instances use only always-supported literals (`bill_shape=curved|0`, `bill_color=yellow|0`, both p=1). Per-literal marginals at K=4 (T): `yellow|1` 0.000, `curved|1` 0.033, `tail:black|1` 0.087, `breast:black|1` 0.167 … while negative versions of the same concepts run 0.86–1.00. The entire K=4 premise failure is one positive literal plus two near-zero positives; the passing set is absent-literals only.

Direction: **above**, not below — premise errors are positively correlated (consistent with the committed var-ratio 5.3 overdispersion in `guarantee_scaling_cub.json`). Compounding *overpredicts* the decay. At K=200 the marginals spread out (95% of 56 distinct literals strictly between 0 and 1, none at 0) and the ratio is consistent with 1.

## Q3 — is 0/118 everywhere invariance or a floor?

Floor. `m1_frac_teacher_sufficient` is saturated (0.0 on all three banks) and separates nothing. The **graded** committed metrics do separate banks: `m2_inreason_agreement` 0.612 / 0.596 / 0.720 and `m3_teacher_necessity` 0.331 / 0.203 / 0.297 (predicates / cls-PCA / meanpatch-PCA, `bankexp/results/comparison_banks_K4.md`), plus X2's completion-based `q(R)` = 0.215 at K=4-T. Use graded metrics in the paper; drop the 0/118 count from any headline. n: 118 (K=4), 243 (K=10), 600 audited (K=200).

## Q4 — present vs absent after base rates (computed 2026-11-11)

Baseline = the concept's occurrence base rate over in-reason occurrences (P(gt=1) for that concept across polarities), not P(gt=1 | asserted), which is the precision itself.

| block | present precision vs base | absent precision vs base |
| --- | --- | --- |
| T / K=4 | 0.154 vs 0.122 (+0.03) | 0.871 vs 0.810 (+0.06) |
| T / K=10 | 0.227 vs 0.173 (+0.05) | 0.914 vs 0.860 (+0.05) |
| T / K=200 | 0.328 vs 0.198 (+0.13) | 0.895 vs 0.825 (+0.07) |

Most of the 0.87 absent precision is the prior. Yellow-bill "bug" question: `yellow_bill_audit.json` shows the reader fires 88/118 with **zero** GT positives in the entire K=4 test set and only **3.3% yellow pixels** in the GT beak box when firing (fires without yellow pixels → shortcut, not threshold artifact), while `certified_only_cub.json` shows the same concept passing all three specificity controls and CUB's own species text saying "buff" (label granularity unresolved, Q5). Presented as the worked example of Q2's concentration finding, not as a standalone claim.

## Q5 — label noise

**No numbers exist.** No noise-quantification run anywhere in the repo; only the yellow/buff case evidence. Candidate run (doubles as the missing Q7 control): inject controlled label noise into GT bits, measure P(all) decay.

## Q6 — reasoning shortcuts / concept leakage (Marconato et al.)

The distinguishing numbers are committed: oracle selection gain 0 (`cbm_cem_audit*.json`), X2 coverage@0.05 = 0.0 for uncertainty-aware reasons, trust filters P(kept-true | covered) 0.067–0.10 (`guarantee_scaling_cub.json`), certified-only retrain leaves P(all) unchanged (`certified_only_cub.json`). Together: *the certificate inherits the premise failure and no selection rule escapes it*. The written comparison against reasoning-shortcut / leakage literature does not exist yet (citations scheduled Nov 15).

## Q7 — the positive control

Conceded: gtbits 0.805 (`design_b_v2.json`) is an **accuracy** on the ladder; premise support with GT bits is trivially 1; and the tree-on-GT-bits control is degenerate at K=4 (`tree_gt_bits_acc` 0.254 = majority). There is no committed positive control for the headline quantity (premise support). The Q5 label-noise run would supply one.

## Q8 — the constructive half (computed 2026-11-11)

Filter reasons to literals whose **marginal annotation support** (estimated out-of-half, cross-fitted) ≥ t:

| block | t=0.5 | t=0.7 | raw P(all) |
| --- | --- | --- | --- |
| D / K=10 | **cov 13.6%, P(all|cov) 0.70** | cov 7.0%, 0.65 | 0.124 |
| T / K=10 | cov 9.1%, 0.64 | cov 2.1%, 0.60 | 0.115 |
| T / K=200 | cov 1.3%, 0.125 | cov 0 | 0.017 |
| K=4 (both) | cov 25.4%, 1.0 (recovers the supported set exactly; abstention, no gain) | same | 0.254 |

**X2's coverage@0.05 = 0 does not kill this** — X2 and the trust filters select on *reader confidence*, and reader confidence does not identify the bad literals; marginal support does. That mismatch (the information needed to abstain exists, but not in the channel people calibrate) is the constructive finding, with the honest scaling limit: nothing survives K=200.

Caveats: marginals estimated on in-reason occurrences (tree-conditioned); cross-fit halves are still test instances. A train/val-estimated version needs a small audit rerun with a marginal dump — the one cheap next run.

---

## What survives for the paper

1. The claim ladder and the transfer-rung framing (drop "100% verified" as a finding; keep as sanity).
2. Premise failure is **concentrated and structured**: one literal carries all of K=4; errors cluster above independence (1.5× at K=10); polarity-asymmetric (positive literals near 0, negative near 1, beyond base rates).
3. The failure is **identifiable** from literal-reliability statistics that reader confidence does not proxy — convertible to a coverage–support tradeoff at K≤10 (70% at 14% coverage), collapsing at 200 classes.
4. Vocabulary ceiling (0.63 / 0.75 / 0.81) and conditional fidelity (0.81 / 0.61) as executed context rungs.

The "passes through all datasets" phrasing is retired: the cross-dataset part that remains is the *decomposition* (exact machinery / failing premises), stated as scope, with the graded bank metrics (Q3) carrying the generalisation evidence.

---

## Re-audit 2026-11-11 (later same day) — Q1 counts/CIs + the Q2 calibration check falsifies the "wrong channel" claim

New artifact: [`abstention_curves.json`](../../grounded_vision/results/abstention_curves.json) via [`scripts/abstention_curves.py`](../../grounded_vision/scripts/abstention_curves.py). Joins the X2 per-instance dump (`min_conf_ord_cal` = min over reason literals of **val-isotonic per-concept calibrated** confidence; `min_conf_ord_raw`) with the audit's per-literal annotation support, by sid, seed 42, reason identity checked. Scores: A_ir = min in-reason marginal support (test-estimated, oracle-flavoured), A_cq = min per-concept accuracy (polarity-blind), B = calibrated min-conf (deployment-honest), C = raw min-conf, D = random.

**Q1 — counts and matched baselines.** The cited operating point (D/K=10, 13.6% coverage) is **33 instances, 23 correct, 0.697 [0.52, 0.85]**; the t=0.7 point is 17 instances. Matched-coverage comparisons at 13.6% (D/K=10): random 0.123 [0.03, 0.24]; raw-confidence 0.030 [0.00, 0.12]. The earlier "6× at matched coverage" framing (vs raw rate at 100% coverage) is withdrawn.

**Q2 — verdict: A ≈ B. The channel claim collapses; the story is miscalibration.**

| block, cov 13.6% | A_ir | A_cq | B_cal | C_raw | D |
| --- | --- | --- | --- | --- | --- |
| T/K=4 | 1.000 | 1.000 | 1.000 | 0.938 | 0.254 |
| D/K=4 | 1.000 | 1.000 | 1.000 | 0.500 | 0.254 |
| T/K=10 | 0.667 [0.52,0.85] | 0.000 | **0.727 [0.52,0.85]** | 0.121 | 0.115 |
| D/K=10 | 0.727 [0.52,0.85] | 0.061 | **0.697 [0.52,0.85]** | 0.030 | 0.123 |

Val-isotonic per-concept confidence recovers the same coverage–support tradeoff as the oracle marginal filter (indistinguishable CIs; B edges A on T/K=10). Raw confidence is at or **below** random at K=10 (anti-informative), and polarity-blind concept accuracy scores ~0 — **polarity-aware per-concept calibration is the load-bearing detail**. The corrected constructive claim: premise failure is concentrated in identifiable (concept, polarity) slots that *standard per-concept calibration on a small val split* exposes; calibrated abstention over reason literals buys the tradeoff, raw confidence does not.

**Open reconciliation (must resolve before the paper):** `guarantee_scaling_cub.json` trusted-only reports P(kept-true | covered) ≈ 0.07–0.1 with per-instance isotonic trust — apparently contradicting B. Protocol difference identified: that rule keeps literals at fixed δ and requires the kept subset to *force* the prediction (a trusted-sufficient-reason query), whereas B is a ranking swept over coverage. Both can hold; state the distinction explicitly or rerun the scaling trust readout as a ranking.

**Not yet run (the run list for Nov 11–16):** Q3 coverage of "sufficient reason over reliable literals only" (the exact query where the formal machinery is non-trivial; needs tree + path pools → local audit rerun); Q4 K=4 subset sensitivity (≥5 random 4-class draws); Q6 mid-K (20, 50) abstention curves — if B degrades gracefully, the method claim stands; if it cliffs, the collapse-at-scale becomes the claim with the scaling curve as a figure; B at K=200 (no X2 K=200 block exists — the calibrated arm has not been measured at scale).

---

## Keep list (final cut, 2026-11-11) — the parts consistent with committed data

User decision after the second review round: not everything above runs; the paper keeps the following, each line checkable against an artifact.

**Spine:**
1. Premise support binding + scaling: 25.4% (K=4, n=118) → 11.5% (K=10, n=243) → 1.7% (K=200, 600/5,794); joint 0.8% at K=200; always < accuracy (0.60/0.44/0.19). [`cbm_cem_audit.json`, `cbm_cem_audit_K200.json`, `guarantee_scaling_cub.json`]
2. Concentration + polarity asymmetry: one literal in all 88 K=4 failures (0/88 support, 3.3% yellow pixels); 30 passes = absent-literals only; present-precision lift over base +0.03…+0.13, absent +0.06/+0.07. K=4 phrased as "this 4-class subset" until subset-sensitivity runs. [`independence_gap_cub.json`, `yellow_bill_audit.json`]
3. Clustering above independence, T/K=10 only, hedged: 1.52 [1.23, 1.89], cross-fit 1.48; D/K=10 and K=200 n.s. → appendix. [`independence_gap_cub.json`]
4. Constructive — miscalibration, not wrong channel: calibrated min-conf abstention at 13.6% coverage D/K=10 = 0.697 [0.52, 0.85], n=33/23 correct; random 0.123; raw conf 0.030 (≤ chance); matches oracle filter 0.727. Val-fit calibrator, test-evaluated. [`abstention_curves.json`]
5. No selection escapes: oracle gain 0 (5/200 MLP); UAR≡ORD 722/722; single-clause negative-literal survivors 25.4%→≤2%. [`cbm_cem_audit*.json`, `x2_uncertainty_reasons.json`]

**Rungs:** 6. ceiling ladder as accuracy (0.627/0.746/0.805); 7. conditional fidelity 0.814/0.613 + margin interventions R≈random; 8. stronger-reader null (0.602→0.678 acc, P(sup) 25.4→24.6%); 9. tree-at-GT-bits collapse (0.254/0.03); 10. graded bank metrics (0.61/0.60/0.72).

**Dropped:** "100% verified" as finding; cross-dataset-invariant phrasing; CLEVR premise claims; 0/118 count; gtbits as premise-support control; "6×"; "no calibrated filter survives K=200" (unmeasured); yellow-bill as standalone claim.

**Four-sentence spine:** (1) premise support, not accuracy or verification, is binding and collapses with scale; (2) failure concentrated in identifiable (concept, polarity) slots; (3) it is miscalibration — calibrated abstention restores 70% at 14% coverage, matching oracle, where raw confidence is chance-level; (4) no selection rule or stronger reader moves it; calibration-gated abstention moves it, at the cost of coverage.
