# Concept Learning

A repository for research in interpretable, verifiable, and causal concept-based machine learning.

---

## 📚 Documentation & Research Wiki

Comprehensive research documentation, mathematical verification proofs, visual evidence galleries, and benchmarks are available in the [**Research Wiki**](docs/wiki/):

- 📖 [**Activation Grounding and Contrastive NLX Wiki**](docs/wiki/ACTIVATION_GROUNDING_WIKI.md): Certified abductive & contrastive reasons, 4×4 spatial pooling over frozen DINO activation maps, and TreeSHAP vs. formal sufficiency comparison.
- 🔍 [**Grounded NLX Examples**](docs/wiki/NLX_EXAMPLES.md): Concrete natural language explanations grounded strictly in verified tree conditions and anatomical CUB part locations.
- 📊 [**Quantitative Benchmark Results**](grounded_vision/results/): Full metric evaluations for activation trees (`activation_tree_grounded_metrics.json`), TreeSHAP comparisons (`shap_vs_sufficient_results.json`), and grounded sufficient reasons (`grounding_selected_reasons.json`).
- 🖼️ [**Visual Evidence Gallery**](docs/wiki/assets/): Visualizations showing verified orange evidence tiles (`activation_tree_grounded_examples.png`) and prototype exemplar matches (`activation_exemplar_examples.png`).

---

## Projects

### 1. Grounded Vision: Certifiable Natural Language Explanations (`grounded_vision/`)
A framework for grounded relational vision and formally verified natural language explanations on fine-grained image benchmarks (e.g. CUB-200-2011).

- **Frozen DINOv2 Backbone**: Localized patch token extraction (`dinov2_vits14`).
- **Perception Subsystem**: Anatomical part detection, region-exemplar prototypes, and bound attribute readers.
- **Relational Predicate Grounding**: First-order logic schema with explicit spatial and exemplar provenance (`PartOf`, `Above`, `Below`, `AdjacentTo`).
- **Certifiable Reasoning**: Interpretable decision tree classifier enforcing taxonomic domain theories and mutual exclusion constraints.
- **Formal Verification**: Independent tree sufficiency verifier & PyXAI minimal prime implicants.
- **Faithful Language Verbalization**: Grounded NL explanation templates with automated audit checking for omitted and hallucinated claims.
- **TreeSHAP Benchmark**: Quantitative attribution comparison against TreeSHAP (88.6% mean overlap with minimal sufficient reasons).

See [`grounded_vision/README.md`](grounded_vision/README.md) for architecture, installation, and pipeline commands.

---

### 2. Concept VAE & Rotated Benchmarks
Variational autoencoder implementations exploring disentangled and rotated concept representations.