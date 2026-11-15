# Results summary — every executed number, one page

**Compiled 2026-11-15** from committed JSONs only (`grounded_vision/results/`). Conventions: official splits; Setting T unless marked; Wilson or cluster-bootstrap (10k over instances) intervals; every rate beside its n. This is the paper's numbers ledger — the drafts cite from here.

---

## 1. Premise support vs accuracy — the inversion (CUB, main audit)

| K | n | acc | P(all\|R) | joint | independence null | ratio [CI] | mean \|R\| |
|---|---|---|---|---|---|---|---|
| 4 | 118 | .602 | **.254** [.184,.340] | — | .254 | 1.00 (degenerate) | 3.6 |
| 10 | 243 | .444 | **.115** [.081,.161] | — | .076 | 1.52 [1.23,1.89] | 5.9 |
| 20 | 515 | .416 | **.070** [.051,.095] | .037 | .061 | 1.15 [0.87,1.44] | 7.1 |
| 50 | 600 | .303 | **.035** [.023,.053] | .022 | .034 | 1.02 [0.64,1.40] | 8.6 |
| 200 | 600 | .193 | **.017** [.009,.031] | .008 | .013 | 1.24 [0.56,2.00] | 10.8 |

Support < accuracy at every K; ~halves per step; rides the null (only K=10 clusters above it). Sources: `cbm_cem_audit.json`, `cbm_cem_audit_K200.json`, `task1_cbcem_K20_50.json`, `independence_gap_K20_50.json`, `guarantee_scaling_cub.json`.

## 2. Concentration — where failure sits

- **CUB K=4:** all 88 failing reasons cite `beak:bill_color=yellow` present, annotation support **0/88**; reader fires on beaks averaging **3.3% yellow pixels** (Fig-example sid 44: **0%**). All 30 passing reasons all-negated; **30/30 survive the strict** unobserved-excluded rule. Positive-literal precision beats base rates by only +0.03…+0.13 (negated +0.06/+0.07). — `independence_gap_cub.json`, `yellow_bill_audit.json`, `cbm_cem_audit.json`.
- **Five re-draws (seeds 1000–1004):** P(all) **0.267–0.431** (none >50%); top literal in **0.62–0.94** of failing reasons but carriers are **negated** literals with marginal support **0.90–0.98** → verdict **SUBSET-SPECIFIC (0/5 dominated)** under the pre-registered criterion. — `task2_subset_sensitivity.json`.
- **CelebA:** top carrier `High_Cheekbones=0` in **1,027/2,042 = 50%** of failing reasons. — `celeba_audit_smiling.json`.
- **BDD-OIA:** top fault-weighted slot `green_light=1` — **1,498 occurrences, support .615**; most *contained* literal `rider=0` in **2,288/2,996 = 76%** of failing reasons at support **.918** (stereotyped co-resident, not the fault). — `bdd_oia_audit.json` (`reasons` key).
- **Clustering vs independence:** CUB 1.52 [1.23,1.89] (K=10) — the only significant departure; BDD **1.003** [1.00,1.01] (at independence); K=4 degenerate. — `independence_gap_cub.json`, `bdd_oia_experiments.json`.

## 3. Fidelity — partial, not binding

Conditional fidelity **0.814** (K=4) / **0.613** (K=10); joint (support ∧ teacher-agreement) **0.037/0.022/0.008** at K=20/50/200 — governed by support. Margins: retention **−6.71** vs −6.25 random, necessity **−0.39** vs −0.55 (R ≈ random); discrete interventions at chance floor (0.246 vs 0.280 donor; 0.254 vs 0.263 meanpatch). Pipit pattern: fidelity 1.0, n=30. — `conditional_fidelity.json`, `design_b.json`, `design_b_v2.json`.

## 4. Vocabulary ceiling — CUB vocabulary-limited, BDD perception-limited

| head input | CUB (K=4) | BDD (forward) |
|---|---|---|
| predicted bits | 0.627 | 0.716 |
| pooled region features | 0.712 | 0.753 |
| continuous scores | 0.746 | — (not run) |
| ground-truth bits | **0.805** | **0.986** |

GT-bits **CART** collapses to majority (0.254/0.03 at 4/200) — a different head family, quoted separately. BDD per-action: stop 0.747/0.755/0.996; left 0.730/0.746/0.913; right 0.685/0.731/0.979. — `design_b_v2.json`, `bdd_oia_experiments.json` (stage6).

## 5. The gate — calibrated weakest-literal abstention

**CUB verdicts (pre-registered bar: bootstrap LB of support-multiple ≥ 2):**

| K | cov | support\|issued | LB ratio | verdict |
|---|---|---|---|---|
| 10 (D) | .136 | 0.697 [.52,.85] (n=33) | vs oracle 0.727, raw 0.030, random 0.123 | B ≈ oracle; raw anti-informative |
| 20 | .14 | 0.361 [.24,.47] (n=72; R=.070) | **3.4** | **cleared** |
| 50 | .14 | 0.143 [.07,.23] (n=84; R=.035) | **2.0** | at the bar |
| 200 | .14 | 0.095 [.04,.15] (n=84; R=.017) | 2.4 | cleared here only |
| 200 | .10 | 0.117 [.03,.22] (n=60) | 1.8 | not cleared |
| 200 | .05 | 0.100 [.00,.23] (n=30) | ≤1 | not cleared |

K=200 rule-3 verification clean (join exact, 0 structural-only reasons, 95% of marginals strictly in (0,1) — the *oracle* filter is dead while calibration still separates). — `abstention_curves.json`, `task1_abstention_K{20_50,200}.json`, `task1_verdicts.json`.

**Replications:** CelebA B **0.957** [.942,.968] @14.2% (raw 0.946, random 0.693); BDD B 0.755 / raw **0.817** @13.6% (raw ≈ calibrated → reader-calibration diagnostic). Discrete reliability channel (A): no separation on any bed. — the two audit JSONs.

**Aggregator ablation (pre-registered P-A1):** min weakly best — @5% cov K=10: min **0.667** vs product 0.583 vs mean 0.583; LB ratio @14%: **4.70 / 4.47 / 3.36**; rank-correlation sub-prediction falsified (Spearman 0.62 < 0.9 at K=10). K=200 aggregator unmeasured. — `ablation_aggregator.json`, `task3_forcing_sweep.json`.

**Forcing ≡ ranking:** the stricter trusted-literals-must-force gate accepts identical instances at every δ∈{0.5..0.9}, K∈{4,10}; the committed fixed-δ read-out's 0.067–0.133 kept-true rates are **raw-channel selection death** (calibrated channel at matched 0.254 coverage: **30/30**). — `task3_forcing_sweep.json`, `guarantee_scaling_cub.json`.

## 6. Nulls — nothing else moves support (CUB; pre-registered)

| intervention | effect | source |
|---|---|---|
| oracle concept/reader selection | gain **0** (best of 200, 5/200) | `cbm_cem_audit.json` |
| stronger reader (MLP) | acc .602→.678; support .254→.246 | `cbm_cem_audit_reader_mlp.json` |
| certified-only retraining | unchanged (P(all) 25.4%; bill-color polarity error persists 0/88) | `certified_only_cub.json` |
| uncertainty-aware reasons | identical **722/722** (BDD: 98.3%) | `x2_uncertainty_reasons.json`, `bdd_oia_experiments.json` |

## 7. BDD-OIA transfer — escapes replicate, training-side nulls reverse

Escapes: oracle subset-selection gain **0.0003**; UAR ≡ ordinary 98.3%; trusted query 0.090 @full coverage (inside CUB's 0.067–0.10 band). Reversals: stronger readers **0.534→0.628**; certified-vocabulary retraining **→0.786** (certificate population reshapes: forward n 2,268→3,383, len 4.0; stop collapses to len→1); GT-bits tree **no collapse** (0.987/0.986). Positive control: i.i.d. annotation noise ε∈[5,30]% moves support within **0.002** of the analytic prediction at every ε. — `bdd_oia_experiments.json`.

## 8. Ablations (pre-registered P-D1..D3)

Depth sweep at fixed K (readers/seed fixed):

| K | depth 2 | depth 12/16 | reading |
|---|---|---|---|
| 4 | .254 (|R| 1.75) | .254 (|R| 3.61) | **flat** — the yellow-bill literal rides every tree |
| 10 | **.358** (|R| 2.0) | .115 (|R| 6.0) | **3.1× ≥ pre-registered 3× bar** |
| 20 | **.528** (|R| 2.0, acc .184) | .070 (|R| 7.2, acc .42) | null predicts **.522** — within 1% |
| 200 | .672 (acc .000) | .028–.038 (acc .16) | trustable reasons from a head that can't classify |

K=20 cells: sid-order subsample — levels not comparable to §1 (trend is the measurement). Ratios to null in [1.00, 1.56] everywhere (P-D2 ✓); |R| monotone in depth (P-D1 ✓). — `ablation_depth_support.json`, `ablation_depth_K20.json`.

## 9. Replication beds (single seed, §4.1 caveat)

| bed | readers | acc | P(all\|R) | n |
|---|---|---|---|---|
| CelebA (Smiling, identity-disjoint) | 0.890 | 0.879 | **0.693** [.682,.704] | 6,654 |
| BDD-OIA (rsbench, pooled) | 0.883 | 0.69–0.75 | **0.534** [.522,.546] | 6,433 |
| SUN (carved split, appendix) | 0.952 | 0.917/0.769 | **0.833**/0.704 (K=4/10) | 60/108 |

Reader-strength gradient: 0.71 / 0.88 / 0.89 / 0.95 reader accuracy ↔ 35 / 20 / 19 / 8-pt support gaps. — `celeba_audit_smiling.json`, `bdd_oia_audit.json`, `sun_audit.json`.

## 10. Methods & model families (BDD-OIA; supplement)

Methods (n=6,423 shared): minimal sufficient reason **2.68 lits / P(all) 0.534 [.521,.546] / sufficiency 1.00 / teacher-flip 0.64**; greedy erasure 0.67 lits / 0.699 / **0.00** / 0.30; top-3 importance 3.00 / 0.578 / 0.37 / 0.59; random-3 3.00 / **0.682** / 0.01 / 0.31 (random wins P(all) via easy negations — 11% positive-literal mix; sufficiency forces rare positives, 27%). Families: joint MLP 0.759 acc but 0.807 post-hoc student agreement; independent CBM worst everywhere (P(all) 0.513); sequential 0.721/0.891/0.542 — binding gap 14–19 pts below each family's own accuracy. — `bdd_oia_methods.json`.

## 11. Engine & machinery verification (sanity tier, never claimed as findings)

Native ≡ PyXAI: **118/118** reasons, **472/472** sufficiency checks (four PyXAI 2.0.1 bugs isolated). PCA-bank control: exact ~3-literal AXps on all banks; teacher-sufficiency **0/118** on all three (feature-vector-invariant); graded agreement 0.612/0.596/0.720. Tree-model pyxai runs locally; `minimal_sufficient_reason` deadlocks on arm64 macOS (bug ii-b; the deferred min-cardinality ablation). — `app:pyxai` ledger, `bankexp/results/comparison_banks_K4.md`.

## 12. Pre-registered task verdicts (2026-11-14 runs)

Task 1: **K=20 GRACEFUL, K=50 GRACEFUL** (rule 1 fired → abstract unchanged), K=200 nominally GRACEFUL → **rule 3 fired, verification clean, fragile** (only the 14% point clears 2×R). Task 2: **SUBSET-SPECIFIC 0/5**. Task 3: **forcing ≡ ranking; 0.07–0.10 = selection death, not support death**. Ablations P-D1/P-D2/P-D3 **all pass** (D3 at 3.1× vs bar 3×); P-A1 outcome ✓ / Spearman ✗. — `task1_verdicts.json`, `task2_subset_sensitivity.json`, `task3_forcing_sweep.json`, `CVPR2027_PREREG_TASK_RUNS.md`.
