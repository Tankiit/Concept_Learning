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
   - **Delivery status 2026-11-14:** branch `distillationAndTransforms` (RSMoon, Nov 13) reviewed — patch rotation/whitening + tree distillation code with tests (9/9, suite 54 pass), a raw/rotated/whitened comparison harness, but ZERO result files; Tasks 1–3 still unexecuted anywhere; merge flags recorded (canonical config default flipped to whitened — must revert to raw)

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
12. [**Behavior-Explainer Comparison (Canizales et al.)**](BEHAVIOR_EXPLAINER_COMPARISON.md)
    - Code audit of CSU-TrustLab/behavior-explainer (arXiv:2605.06640): their AXp oracle IS the neural head via CLIP-erasure (fidelity by construction, exactness conditional on monotonicity 52.5–99.1%), 42–10,234 AXps/image vs our unique 2.89-literal reason
    - Dataset verdicts (RIVAL10/EuroSAT: premise support unmeasurable) + ready related-work paragraph for the Nov 15 citation pass

13. [**Second Datasets — Survey, Downloads, Three Executed Audits**](SECOND_DATASETS.md)
    - SUN / CelebA / DeepFashion / RIVAL10 / EuroSAT: provenance ledger, verification, base rates
    - EXECUTED rows: CelebA 0.693 (n=6,654), BDD-OIA/rsbench 0.534 (n=6,433), SUN 0.833/0.704 (K=4/10, carved split) — premise support below accuracy on every bed
    - The reader-strength gradient (readers .71/.88/.89/.95 → gaps 35/20/19/8 pts) + three-dataset abstention verdict

14. [**Paper Intro Draft (v3, number-free)**](PAPER_INTRO_DRAFT.md)
    - LaTeX intro built from the keep list; number-free per user choice; quantity→artifact map for the body; changelogs v1→v3

15. [**Paper Body Draft (v2, CVPR-style)**](PAPER_BODY_DRAFT.md)
    - §2 related work · §3 method (formal definitions, claim ladder, eqs) · §4 experiments (CVPR-style rewrite) · §5–6 + appendix stubs (PyXAI ledger, subset, clustering, optional SUN)
    - Per-table provenance map to committed JSONs, drop-list compliance, freeze flags, abstention figure script
