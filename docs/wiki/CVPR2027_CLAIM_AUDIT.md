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

---

## Addendum 2026-11-12 (evening) — off-CUB replications executed (CelebA, BDD-OIA/rsbench); status relative to the keep list

Not part of the 2026-11-11 final cut; produced after it, same evening, to answer the "single dataset" objection. Artifacts `results/celeba_audit_smiling.json` (n=6,654 reasons) and `results/bdd_oia_audit.json` (n=6,433) via `scripts/celeba_audit_stage_*.py`, `scripts/rsbench_bdd_audit.py` — **uncommitted as of writing; commit before the paper cites them.** Protocol: seed 42 only, single depth-4 tree per target, official splits, Wilson CIs, matched-random abstention baselines; CelebA uses our DINOv2 features, BDD-OIA uses the rsbench preprocessed 2048-d features (their pipeline — grounds the Teso/Marconato comparison). Treat as **replication evidence**, not headline claims; CUB rows above remain the audited main results.

**Spine sentence 1 (binding) — replicates, add two rows:** CelebA (binary Smiling, readers 0.890): acc 0.879/0.875, P(all)=0.693 [0.682, 0.704], reason len 2.02. BDD-OIA (5 actions, readers 0.883): teacher/tree 0.69–0.75, pooled P(all)=0.534 [0.522, 0.546], per-action 0.472/0.596/0.198/0.558. Both ≈20 points below accuracy — "explanation less likely fully right than prediction" now three-way.

**Spine sentence 2 (concentration) — replicates; polarity direction is DOMAIN-SPECIFIC:** CelebA failing slots are *negative* (High_Cheekbones=0 0.799 n=3,214; Mouth_Slightly_Open=0 0.823 n=2,749); BDD-OIA failing slots are *positive* traffic literals (green_light=1 0.615 n=1,498; red_light=1 0.522; follow=1 0.436; stop_sign=1 0.386) with negatives at ceiling (clear=0 0.931) — matching CUB's pattern. The transferable claim: identifiable (concept, polarity) slots; "positive literals" is CUB+BDD, not universal.

**Spine sentence 3 (miscalibration) — RE-SCOPE:** min-confidence abstention lifts support on all three datasets (CUB 0.697@13.6%; CelebA 0.957@14%; BDD 0.723@5.5%); the *calibration-specific* superiority is CUB-only (CelebA raw≈cal 0.946; BDD raw>cal 0.817 vs 0.723@5%). Discrete per-(concept,polarity) reliability fails everywhere (BDD 0.581 plateau; CelebA 0.628@8.8% < random). Paper wording: recommend calibrated min-conf universally; claim "raw is chance-level" only for CUB (where miscalibration is demonstrated).

**Scope statement change:** "one dataset, stated as scope" is superseded for the premise-support measurement (now three datasets); CUB-only quantities unchanged (K-scaling, interventions, ceilings, no-selection-escapes).

Open before submission: single seed/depth for the replications (state in body); commit artifacts; decide main-paper table vs rebuttal (recommendation: main paper, one compact table + one paragraph — it removes the top structural objection; SUN may add a third row, wording kept count-agnostic).

## Addendum 2026-11-13 — BDD-OIA experiment suite executed (full CUB rungs on rsbench)

`scripts/rsbench_bdd_experiments.py` → `results/bdd_oia_experiments.json` (uncommitted; stage 1 reproduces the audit to Δ −0.0006). Full deciding table in [`SECOND_DATASETS.md`](SECOND_DATASETS.md). Consequences for the keep list:

- **Spine 5 strengthened, split in two:** *certificate-level* escapes replicate on their benchmark — oracle subset gain 0.0003, UAR ≡ ORD on 98.3% (q gain 0.0014), full-vector fixed-δ trust P(kept-true|covered) 0.090 at coverage 1.0 (CUB: 0.067–0.10). *Training-level* nulls do **not**: MLP readers lift pooled P(all) 0.534 → 0.628; val-acc-certified vocabulary lifts it to 0.786 (12 concepts, certificate population reshapes — forward n 2,268→3,383, stop → len-1 reasons, right n→137; accuracy roughly preserved). The sentence-4 clause "no selection rule **or stronger reader** moves it" must become CUB-scoped: on BDD-OIA the failure is perception-limited (GT-bits teacher 0.95+, GT-bits tree does *not* collapse: 0.987/0.986 forward/stop), so perception-side investment moves it; on CUB it is vocabulary/annotation-limited and nothing moves it. This split *is* the finding — the audit decomposes failure into vocabulary vs premises, and the two datasets land on opposite sides.
- **Spine 3 stays CUB-only and gains a negative control:** BDD-OIA observed/independence = 1.003 [1.00, 1.01] (right action 0.948, below 1) — premise errors at independence, no clustering. Keep 1.52 [1.23, 1.89] hedged in the appendix, now with an explicit "not universal" row.
- **Spine 2 wording:** concentration is identifiable-but-distributed on BDD (top slot `green_light=1` in 22.5% of failures, top-3 in 78.4%; CUB K=4 was 100% one literal). Present-precision lift over base is +0.32 (0.561 vs 0.241) — the failure there is the near-zero base rate of traffic concepts, not precision collapse.
- **Spine 4 unchanged, with two new supports:** calibrated mean q of ORD reasons 0.5349 vs observed P(all) 0.5337 — the isotonic completion model is honest on BDD; reason-scoped trusted ranking reaches 0.717 at 22.6% coverage. Q8-style marginal-threshold filters cover 0 cross-fitted on BDD (they were CUB-flavoured); the transferable tool remains calibrated min-conf ranking.
- **Q5/Q7 positive control delivered (on BDD):** iid GT-bit flips at ε = 5–30% give P(all) 0.485/0.439/0.355/0.279 vs iid-analytic 0.484/0.438/0.354/0.281 — the metric responds exactly as an annotation-noise model predicts. The CUB-side noise run remains open (cheap: same protocol on committed per-instance dumps).
- **Methods/models comparison (2026-11-13, same evening):** `scripts/rsbench_bdd_methods.py` → `results/bdd_oia_methods.json`. Other explanation methods on the frozen pipeline: only certified reasons are sufficient (forces 1.0; erasure 0.0 with 0.67 kept bits, top-3 importance 0.37, random 0.01) and most teacher-faithful (0.635); random matched-size literals score *higher* P(all) (0.682 vs 0.534) purely from easy absent-literals — always report assertion-mix next to P(all). Their model families retrained on their tensors: independent CBM worst on every column (acc 0.707 / exact-4 0.316 / P(all) 0.513 — the RS deployment penalty under our audit), joint MLP best acc (0.759) but post-hoc student agrees only 0.807, binding gap uniform 14–19 points across families. Full tables in [`SECOND_DATASETS.md`](SECOND_DATASETS.md).
