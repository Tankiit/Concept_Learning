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
   - One-liner, 5-minute arc, and the full five-act narrative
   - The claim ladder (certified / grounded / not yet claimable) and anticipated Q&A
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

9. [**Visual Evidence Gallery**](assets/)
   - [Activation Tree Grounded Evidence Tiles (train/test split)](assets/activation_traintest_examples.png)
   - [Activation Test→Train Exemplar Retrieval](assets/activation_traintest_exemplars.png)
   - [Activation Tree Grounded Evidence Tiles (historical 100-image rerun)](assets/activation_tree_grounded_examples.png)
   - [Activation Exemplar Prototype Matches (historical rerun)](assets/activation_exemplar_examples.png)

10. [**Raw Result Files**](../../grounded_vision/results/)
   - `activation_track_traintest_results.json` / `..._c4.json` (official split, primary)
   - `activation_tree_grounded_metrics.json`, `shap_vs_sufficient_results.json`, `grounding_selected_reasons.json` (historical in-cache protocol, fixed-pooling rerun)
   - `clevr_relational_results.json`
   - `cub_bound_mock_benchmark_results.json`
   - `predicate_grounding_milestone.json`
