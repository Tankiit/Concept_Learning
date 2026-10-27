# Concept Learning

A repository for research in interpretable, verifiable, and causal concept-based machine learning.

---

## Projects

### 1. Grounded Vision: Certifiable Natural Language Explanations (`grounded_vision/`)
A framework for grounded relational vision and formally verified natural language explanations on fine-grained image benchmarks (e.g. CUB-200-2011).

- **Frozen DINOv2 Backbone**: Localized patch token extraction.
- **Perception Subsystem**: Anatomical part detection, region-exemplar prototypes, and bound attribute readers.
- **Relational Predicate Grounding**: First-order logic schema with explicit spatial and exemplar provenance (`PartOf`, `Above`, `Below`, `AdjacentTo`).
- **Certifiable Reasoning**: Interpretable decision tree classifier enforcing taxonomic domain theories and mutual exclusion constraints.
- **Formal Verification**: Independent tree sufficiency verifier & PyXAI minimal prime implicants.
- **Faithful Language Verbalization**: Grounded NL explanation templates with automated audit checking for omitted and hallucinated claims.

See [`grounded_vision/README.md`](grounded_vision/README.md) for full details, installation, and usage.

---

### 2. Concept VAE & Rotated Benchmarks
Variational autoencoder implementations exploring disentangled and rotated concept representations.