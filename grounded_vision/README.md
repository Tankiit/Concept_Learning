# Grounded Vision: Certifiable Natural Language Explanations

`grounded_vision` is a research framework for **grounded, relational vision** and **verifiable natural language explanations** in fine-grained image classification (e.g. CUB-200-2011).

Instead of treating concept bottleneck models as ungrounded bags of attributes or generating unconstrained natural language via LLMs that easily hallucinate, `grounded_vision` enforces a clean, mathematically sound pipeline:

```
Image 
  │
  ▼ [Frozen DINO Backbone]
Patch Embeddings
  │
  ▼ [Perception Layer]
Part Regions + Prototypes + Attribute Readers
  │
  ▼ [Predicates Layer]
Grounded Fact Database (Entities + Predicates + Explicit Provenance + Relations)
  │
  ▼ [Reasoning Layer]
Feature Compiler (Flat / Bound / Relational) ──► Certifiable Decision Tree Head
  │
  ▼ [Explanation Layer]
Formal Tree Verifier / PyXAI ──► Minimal Sufficient Reason (Prime Implicants)
  │                              │
  ▼                              ▼
Visual Evidence (BBoxes, Crops, Prototypes) ──► Faithful NL Verbalizer (Audited)
```

---

## Directory Structure

```
grounded_vision/
├── configs/
│   ├── cub_flat.yaml          # Global flat attribute concepts baseline
│   ├── cub_bound.yaml         # Entity-bound attribute concepts
│   └── cub_relational.yaml    # Relational concepts (bound + PartOf + spatial)
├── src/grounded_vision/
│   ├── data/
│   │   ├── cub.py             # CUB-200 dataset loader & synthetic generator
│   │   └── splits.py          # Train / val / test splitting routines
│   ├── perception/
│   │   ├── backbone.py        # Frozen DINO patch feature extraction
│   │   ├── regions.py         # Part regions, masks, spatial hierarchy
│   │   ├── prototypes.py      # Region–exemplar similarity dictionary
│   │   └── readers.py         # Part presence and attribute predictors
│   ├── predicates/
│   │   ├── schema.py          # Entities, predicates, facts, provenance
│   │   ├── grounding.py       # Reader outputs → calibrated facts with provenance
│   │   └── relations.py       # PartOf and directional spatial relations
│   ├── reasoning/
│   │   ├── features.py        # FactDatabase → structured feature vectors
│   │   ├── tree_head.py       # Interpretable, certifiable classifier
│   │   └── domain.py          # Explicit domain theories & taxonomic constraints
│   ├── explanations/
│   │   ├── schema.py          # Conditions, sufficiency certificates, contrastives
│   │   ├── pyxai_backend.py   # PyXAI integration + standalone exact solver
│   │   ├── tree_verifier.py   # Independent formal sufficiency verifier
│   │   └── evidence.py        # Conditions → regions, masks, and exemplars
│   ├── language/
│   │   ├── templates.py       # Deterministic faithful verbalization templates
│   │   ├── nlx_adapter.py     # Grounded evidence conditioning for NL decoders
│   │   └── checks.py          # Faithfulness auditor (omitted & unsupported claims)
│   └── evaluation/
│       ├── perception.py      # Part localization and attribute metrics
│       ├── classification.py  # Accuracy, F1, calibration (ECE)
│       ├── explanations.py    # Formal sufficiency rate, sparsity, necessity
│       └── interventions.py   # Causal predicate intervention tests
├── scripts/
│   ├── cache_features.py      # Extract and cache DINO patch features
│   ├── fit_perception.py      # Train/fit part and attribute readers
│   ├── export_predicates.py   # Ground perception into fact databases
│   ├── train_head.py          # Train interpretable tree/rule classifier
│   ├── explain.py             # Generate certified explanations for instances
│   └── evaluate.py            # Comprehensive evaluation benchmark
├── notebooks/
│   └── inspect_one_image.ipynb# End-to-end interactive visual walkthrough
└── tests/
    ├── test_predicate_binding.py
    ├── test_classifier_export.py
    └── test_sufficiency.py
```

---

## Quickstart

### Installation

```bash
cd grounded_vision
pip install -e .
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Pipeline Scripts

```bash
# 1. Cache backbone features (or synthetic mock for testing)
python scripts/cache_features.py --config configs/cub_bound.yaml

# 2. Fit perception readers and prototypes
python scripts/fit_perception.py --config configs/cub_bound.yaml

# 3. Export grounded predicate databases
python scripts/export_predicates.py --config configs/cub_bound.yaml

# 4. Train certified decision tree head
python scripts/train_head.py --config configs/cub_bound.yaml

# 5. Explain test samples
python scripts/explain.py --config configs/cub_bound.yaml --sample-idx 0

# 6. Run comprehensive evaluation
python scripts/evaluate.py --config configs/cub_bound.yaml
```
