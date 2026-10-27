# Grounded Vision Wiki & Research Documentation

Welcome to the **Grounded Vision** documentation wiki for certifiable, relational concept learning and natural language explanations.

---

## Table of Contents

1. [**Activation Grounding and Contrastive NLX**](ACTIVATION_GROUNDING_WIKI.md)
   - Certified abductive and contrastive reasons on real CUB-200-2011 images
   - 4×4 spatial pooling over frozen DINO activation maps (6,144 features)
   - 100% verified sufficient-reason rate (mean size: 3.8 conditions)
   - Attribution baseline comparison: TreeSHAP vs. Formal Sufficiency (88.6% overlap)
   - Part-only NLX policy and model-level counterfactual sensitivity

2. [**Grounded Natural Language Explanation (NLX) Examples**](NLX_EXAMPLES.md)
   - Concrete faithful natural language explanation examples across species
   - Evidence tiles linked to validated CUB anatomical part locations

3. [**CUB Real-Data Pipeline Run Report**](CUB_BOUND_REAL_FULL_RUN.md)
   - Full real CUB-200-2011 run configuration (`cub_bound_real_full.yaml`)
   - Feature caching and perception checkpoints

4. [**Visual Evidence Gallery**](assets/)
   - [Activation Tree Grounded Evidence Tiles](assets/activation_tree_grounded_examples.png)
   - [Activation Exemplar Prototype Matches](assets/activation_exemplar_examples.png)

5. [**Quantitative Results & Metrics**](../../grounded_vision/results/)
   - `activation_tree_grounded_metrics.json`
   - `shap_vs_sufficient_results.json`
   - `grounding_selected_reasons.json`
