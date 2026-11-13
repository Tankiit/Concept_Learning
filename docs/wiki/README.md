# Grounded Vision Wiki & Research Documentation

Welcome to the **Grounded Vision** documentation wiki for certifiable, relational concept learning and natural language explanations.

---

## Table of Contents

1. [**Grounded Vision Framework — Architecture Reference**](GROUNDED_VISION_FRAMEWORK.md)
   - Full module reference (perception → predicates → reasoning → explanations → language → evaluation)
   - Configuration matrix (8 configs: flat / bound / relational, CUB-28 / CUB-312, CelebA)
   - Setup, testing, and known repository issues (uncommitted `data/` subpackage)

2. [**Experiment Tracks & Script Catalog**](EXPERIMENT_TRACKS.md)
   - All 32 scripts: purpose, inputs, outputs, and where each track can run
   - Core pipeline · activation grounding · reviewer analysis · CLEVR relational · validation milestone · Modal cloud

3. [**Quantitative Results — Consolidated Index**](RESULTS.md)
   - Activation track on real CUB, official train/test split: 56.8% held-out accuracy (4-species), 100% verified reasons on test; 200-class run shows certificates hold (4.3% accuracy — the honest trade)
   - CLEVR relational track: 69.8% accuracy, 100% relational + necessary certified reasons
   - Core-pipeline mock smoke benchmark (recovered from the Modal image)
   - Predicate grounding milestone v1+v2 (real CUB): 82.8% probe agreement; separation persists under deployed readers; crown/breast v1 failure largely explained by adjacent controls
   - Pending server-side artifacts and the results-refresh protocol

4. [**The Research Story & Talk Track**](THE_STORY.md)
   - One-liner, 5-minute arc, and the full story: problem → design principle → the adversarial-review stress test → evidence → claim ladder
   - The claim ladder (proved / measured / sensitivity-not-correction / open) and anticipated Q&A
   - Every number traced to a committed result file

5. [**Review Validation Roadmap**](REVIEW_VALIDATION_ROADMAP.md)
   - External review (2026-11-01): claim-by-claim verification against the code
   - Fixed: verifier class mapping, negative relational templates, tile pooling (regeneration pending)
   - Milestone executed: predicate grounding with perturbation specificity (beak/belly pass, crown/breast open)
   - Open: three-valued predicates, binding-aware audit, binding-benefit study

6. [**Activation Grounding and Contrastive NLX**](ACTIVATION_GROUNDING_WIKI.md)
   - Certified abductive and contrastive reasons on real CUB-200-2011 images
   - 4×4 spatial pooling over frozen DINO activation maps (6,144 features, corrected axes)
   - 100% verified sufficient-reason rate on held-out test images (mean 3.79 conditions)
   - Attribution baseline comparison: TreeSHAP vs. Formal Sufficiency (87.9% overlap on test)
   - Part-only NLX policy and model-level counterfactual sensitivity

7. [**Grounded Natural Language Explanation (NLX) Examples**](NLX_EXAMPLES.md)
   - Concrete faithful natural language explanation examples across species
   - Evidence tiles linked to validated CUB anatomical part locations

8. [**CUB Real-Data Pipeline Run Report**](CUB_BOUND_REAL_FULL_RUN.md)
   - Full real CUB-200-2011 run configuration (`cub_bound_real_full.yaml`)
   - Feature caching and perception checkpoints

9. [**CVPR 2027 Submission Plan — The Eight-Day Cut**](CVPR2027_SUBMISSION_PLAN.md)
   - Verified calendar (registration Nov 10 / submission Nov 16, 2026 AOE — official pages, no extensions)
   - Scope cut with execution status per experiment (X1 ceiling and X3 fidelity+interventions already executed at HEAD `d4e66b5`; X2 uncertainty-aware comparison is the only new "must")
   - Corrections to the source plan: setting T/D and X-numbering rename (A/B and E0/E1 already taken), flip condition not pre-registered (existing pre-regs predict the opposite), related-work citations not in repo
   - Eight-day schedule, reviewer defences evidence-mapped to committed JSONs, abstract skeleton, verification ledger

10. [**CVPR 2027 Claim Audit — Eight Questions Against the Invariant-Claim Abstract**](CVPR2027_CLAIM_AUDIT.md)
   - 2026-11-11 external review verdicts: certificate-exactness is by construction (unit test, not a finding); 0/118 teacher-sufficiency is a floor (graded bank metrics separate banks instead)
   - New artifact `independence_gap_cub.json`: K=4 failure is one literal (bill_color=yellow, 0/88 support, in every failing reason); errors cluster ABOVE independence at K=10 (ratio 1.52 [1.23, 1.89]); present/absent clause precision beats base rates by only +0.03…+0.13 / +0.06
   - Constructive half alive at K≤10: cross-fitted literal-reliability filter gives 70% joint support at 14% coverage (vs 12% raw) on the distilled arm; dead at K=200; reader confidence does not identify the bad literals (X2 coverage@0.05 = 0)
   - Filled title + abstract written into the submission plan §9 the same day

11. [**CVPR 2027 Hand-off — The Three Runs (Assignment Brief)**](CVPR2027_HANDOFF_TASKS.md)
   - Runner-facing assignment as written: Task 1 calibrated abstention at K=20/50 + first-ever calibrated arm at K=200 (due first — decides tradeoff vs collapse), Task 2 five random 4-class draws, Task 3 forcing sweep (explains guarantee_scaling 0.07–0.10)
   - Ground rules (new files only, per-instance dumps, val-only calibrators, n beside every percentage, send Task 1 numbers immediately) + early-finish job: reasoning-shortcuts / Teso comparison table (citations due Nov 15)
   - Links to the ready scripts (grounded_vision/scripts/tasks/), the pre-registered cliff criteria, and the division of labour: numbers to the runner, writing and trust-result reconciliation to the lead

12. [**Visual Evidence Gallery**](assets/)
   - [Activation Tree Grounded Evidence Tiles (train/test split)](assets/activation_traintest_examples.png)
   - [Activation Test→Train Exemplar Retrieval](assets/activation_traintest_exemplars.png)
   - [Activation Tree Grounded Evidence Tiles (historical 100-image rerun)](assets/activation_tree_grounded_examples.png)
   - [Activation Exemplar Prototype Matches (historical rerun)](assets/activation_exemplar_examples.png)

13. [**Raw Result Files**](../../grounded_vision/results/)
   - `activation_track_traintest_results.json` / `..._c4.json` (official split, primary)
   - `activation_tree_grounded_metrics.json`, `shap_vs_sufficient_results.json`, `grounding_selected_reasons.json` (historical in-cache protocol, fixed-pooling rerun)
   - `clevr_relational_results.json`
   - `cub_bound_mock_benchmark_results.json`
   - `predicate_grounding_milestone.json`
