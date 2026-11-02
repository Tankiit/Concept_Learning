# Grounded Vision framework — architecture reference

**Package:** `grounded_vision/` (v0.1.0, src-layout under `src/grounded_vision/`)
**Goal:** grounded, relational concept learning with **verifiable** natural language explanations for fine-grained image classification (CUB-200-2011, CelebA, CLEVR-style scenes).

The framework rejects two common failure modes: concept bottleneck models treated as ungrounded attribute bags, and unconstrained LLM explanations that hallucinate. Every explanation is a formally verified logical condition over grounded facts, each carrying explicit provenance back to image regions.

---

## Pipeline overview

```
Image
  │  Frozen DINOv2 ViT-S/14 (384-d, patch 14, 224 px)
  ▼
Patch embeddings + CLS token
  │  Perception layer
  ▼
Part regions + prototypes + attribute readers
  │  Predicates layer
  ▼
FactDatabase (entities + facts + provenance + relations)
  │  Reasoning layer
  ▼
FeatureCompiler (FLAT / BOUND / RELATIONAL) ──► CertifiableTreeHead
  │  Explanation layer
  ▼
TreeSufficiencyVerifier / PyXAI ──► minimal sufficient reasons (prime implicants) + contrastives
  │  Language layer
  ▼
Visual evidence (bboxes, crops, prototypes) ──► faithful NL verbalizer (audited)
```

The chain is **invertible by construction**: a verbalized sentence maps back to logical conditions, which map back to facts, which map back to regions, prototypes, and reader probabilities.

---

## Module reference

### `perception/` — feature extraction and neural reading

| Module | Responsibility |
| --- | --- |
| `backbone.py` | `FrozenDINOBackbone` wraps a frozen ViT (`dinov2_vits14`) via torch.hub; returns patch tokens, CLS token, and a `[B, D, H, W]` feature map. Falls back to a deterministic mock projector when DINO weights or torch are unavailable (offline/testing mode, exposed via `is_mock`). |
| `regions.py` | `Region` dataclass (part name, normalized bbox, confidence, mask), `SpatialHierarchy` (bird → macro → micro part tree, IoU), `extract_part_regions`, and `pool_region_features` (RoI pooling of patch features, L2-normalized). |
| `readers.py` | `PartDetector` (part presence + bbox heads from the CLS token) and `AttributeReader` (one linear head per bound attribute; whole-bird attributes read the whole-image region). Emits `PerceptionOutput`. |
| `prototypes.py` | `AttributePrototypeBank` — exemplar bank giving provenance of the form "beak is 0.88 similar to curved-bill prototype #3" (cosine, best-of-K per attribute). |

### `predicates/` — grounded first-order facts

| Module | Responsibility |
| --- | --- |
| `schema.py` | The ontology: `Entity` / `BirdEntity` / `PartEntity`, `PredicateType` (`HAS_PART`, `HAS_ATTRIBUTE`, `PART_OF`, `SPATIAL`), `Fact` with canonical keys (e.g. `part_beak_0:bill_shape=curved`), `FactDatabase`, and `Provenance` (region id, bbox, detector confidence, reader probability, prototype id/similarity, patch indices) attached to every fact. |
| `grounding.py` | `ground_perception_output` converts a `PerceptionOutput` into a calibrated `FactDatabase` (entities, `has_part`, `has_attribute`; whole-bird attributes get subject `bird_0`). |
| `relations.py` | `compute_spatial_relation` (`above/below/left_of/right_of/adjacent_to`, margin `delta_coord=0.05`) and `extract_relations` (anatomical `part_of` facts + spatial facts over salient part pairs). |

### `reasoning/` — interpretable classifier over facts

| Module | Responsibility |
| --- | --- |
| `features.py` | `FeatureCompiler` with `RepresentationMode`: **FLAT** = global `attr=val`; **BOUND** = `part:attr=val` + `has_part(part)` bits (canonical 28-attribute vocabulary → 36 features); **RELATIONAL** = BOUND + spatial relation literals. Builds an invertible vocabulary (`.feature_names`, `.feature_to_idx`) and compiles `FactDatabase`s to binary or probability vectors. |
| `tree_head.py` | `CertifiableTreeHead` — sklearn `DecisionTreeClassifier` wrapper with symbolic decision paths (`DecisionPathStep.condition_str`) and `export_rules()` (every leaf path as a symbolic rule). |
| `domain.py` | `DomainTheory` — mutual-exclusion groups (bill shapes/colors, crown/breast/belly/wing colors) and prerequisite checks (attribute requires its parent part). |

### `explanations/` — formal certificates

| Module | Responsibility |
| --- | --- |
| `tree_verifier.py` | `TreeSufficiencyVerifier` — independent exact solver: a condition set is **sufficient** iff at least one tree leaf is reachable and *zero* reachable leaves predict another class (reachability via bound propagation). `extract_minimal_sufficient_conditions` computes a prime implicant by backward elimination along the decision path. |
| `pyxai_backend.py` | `PyXAIBackend` — `explain_sufficient` / `explain_contrastive` (minimal violated conditions to the nearest contrast leaf). PyXAI itself computes nothing here: its d4 compiler is x86-64 Linux only, so the backend delegates to the native verifier and reports `solver = "native_tree_verifier"`. Compiler-based PyXAI proofs run through `scripts/run_pyxai_modal.py` on Linux. |
| `schema.py` | `ExplanationCondition`, `SufficientExplanation` (with `.sparsity`), `ContrastiveExplanation`, `VerificationCertificate`, `VerifiedExplanation`. |
| `evidence.py` | `EvidenceResolver` — each logical condition resolves to a `VisualEvidenceCard` (bbox, crop, confidence, prototype match): the visual grounding of the proof. |

### `language/` — faithful verbalization + audit

| Module | Responsibility |
| --- | --- |
| `templates.py` | `ExplanationTemplateGenerator` — deterministic, polarity-aware templates (`<=` verbalizes as *absence*). |
| `checks.py` | `ExplanationAuditor` — detects omitted conditions (recall) and unsupported/hallucinated claims (color/shape vocabulary checked against certified tokens; precision). |
| `nlx_adapter.py` | `NLXConditioningAdapter` — builds decoder prompts restricted to verified facts, with a deterministic template fallback (`decode(decoder_type="template")`). |

### `evaluation/` — benchmarks

| Module | Responsibility |
| --- | --- |
| `perception.py` | Mean part IoU, PCK, attribute accuracy/precision/recall/F1. |
| `classification.py` | Accuracy, macro/micro-F1, calibration (ECE, 10 bins). |
| `explanations.py` | Sufficiency rate, sparsity stats, NL faithfulness/precision/recall/hallucination rate. |
| `interventions.py` | `do(X_j = 1 - x)` concept-bit flips on top-5 tree features → prediction flip rates. |

### `data/` — dataset loading (committed 2026-11-01, `96e0225`)

| Module | Responsibility |
| --- | --- |
| `cub.py` | `CUBDataset` (real CUB-200-2011 loader with `use_mock_if_missing` synthetic mode), the 15-part anatomical ontology (`CUB_PARTS`, `CUB_PART_HIERARCHY`, `PART_TO_MACRO`), `CANONICAL_BOUND_ATTRIBUTES` (the 28 bound attributes), `CUB_ATTRIBUTES`, `DEFAULT_SPECIES`, `CUBSample`, transforms, and `create_synthetic_cub_batch`. |
| `attributes.py` | Bound-attribute vocabulary registry — makes the vocabulary a config-driven dimension: `get_bound_attributes("canonical" \| "full", data_dir)`, `vocabulary_cub_ids` (aligned raw CUB ids for label supervision), `active_vocabulary(cfg)`, `celeba_vocabulary`, `valid_parts`. "canonical" = the 28 hand-picked part-bound attributes; "full" = all 312 CUB attributes bound to grounding regions by name prefix (wing/leg attributes → both instances; part-less attributes → the whole-bird `bird` entity). |
| `celeba.py` | `CelebADataset` — same interface as `CUBDataset`, so the whole pipeline runs unchanged: the 40 binary attributes bound to 5 landmark-derived part regions (left_eye/right_eye/nose/mouth/face), a held-out target attribute (default `Male`) as the class label and excluded from the concept vocabulary, official splits with configurable subset sizes. |
| `splits.py` | Train/validation/test split generation and verification (`get_cub_split_indices`, `verify_splits`). |

### `pipeline.py` — shared plumbing

`pick_device` (cuda > mps > cpu), `build_backbone`, `load_perception` (restores `perception_checkpoint.pt`; warns and falls back to fresh modules on miss), `FeatureProvider` (serves cached `{split}_features.npz`), `regions_for_sample` (`gt` oracle boxes vs `predicted` PartDetector), `perceive_sample` (image → regions → pooled features → grounded `FactDatabase`), `build_dataset` (CUB or CelebA).

---

## Configuration matrix (8 configs)

All share: seed 42, DINOv2 ViT-S/14, gini decision tree with `enforce_domain_constraints`, `tree_verifier` explanation backend, rule-based scientific decoder, strict faithfulness audit.

| Config | Dataset / cache dir | Experiment | Key differences |
| --- | --- | --- | --- |
| `cub_flat.yaml` | CUB, relative `data/`, `cache/cub_flat` — **mock allowed** | Flat CBM baseline | `representation: flat`, no part binding/relations |
| `cub_bound.yaml` | CUB, relative paths — **mock allowed** | Entity-bound grounding (core condition) | `bound`, part detection on, depth 12, `max_conditions: 6` |
| `cub_relational.yaml` | CUB, relative paths — **mock allowed** | Relational concepts | `relational`, `include_relations: true`, depth 14, `max_conditions: 8` |
| `cub_bound_real_full.yaml` | CUB real (CRIL paths) | Documented real-data run (2026-10-27) | canonical 28 attributes |
| `cub_bound_real_smoke.yaml` | CUB real (CRIL paths) | Smoke test of the local CUB install | same settings, isolated cache dir |
| `cub_bound_full.yaml` | CUB real (CRIL paths) | Full-pipeline rerun with fitted perception | canonical 28; feeds the reviewer-analysis track |
| `cub_bound_312.yaml` | CUB real (CRIL paths) | Full 312-attribute vocabulary | the only `attribute_set: "full"` config |
| `celeba_concepts.yaml` | CelebA (CRIL paths) | Concept-grounded CelebA | 2 classes, 5 landmark parts, 40 attributes, `Male` held out as target |

Vocabulary note: every CUB config declares `num_attributes: 312` as metadata, but the *effective* vocabulary is the canonical 28 bound attributes everywhere except `cub_bound_312.yaml`.

---

## Setup and testing

```bash
cd grounded_vision
pip install -e .          # torch, torchvision, numpy, scipy, scikit-learn, pyyaml, pillow, matplotlib
pip install -e ".[dev]"   # + pytest, jupyter
pip install -e ".[pyxai]" # + pyxai (Linux only; macOS uses the native verifier)
pytest tests/ -v
```

Tests (`tests/`, ~14 cases) run entirely on synthetic in-memory data — no images, backbone, or dataset required. `test_full_vocabulary.py` skips its real-data case unless `$GROUNDED_VISION_CUB_DIR` points at a CUB checkout with `attributes.txt`.

### ✅ Resolved: `src/grounded_vision/data/` is now committed

The `data/` + `**/data/` ignore rules in `grounded_vision/.gitignore` (meant for dataset folders) silently matched `src/grounded_vision/data/`, so the subpackage was **never committed** and every fresh clone failed with `ModuleNotFoundError: No module named 'grounded_vision.data'` (the whole test suite errored at collection). Resolution timeline:

1. **2026-11-01 (stopgap):** `cub.py`, `splits.py`, `__init__.py` recovered verbatim from the deployed Modal image (`im-q3xHgkFSVyRh7LLH6gAJXx`, an Oct-27 snapshot of the server tree) — restored 11/11 tests in three test files.
2. **2026-11-01 (commit `96e0225`):** the full subpackage committed from the dev server — including `attributes.py` and `celeba.py`, which postdate the Modal image (vocabulary work of 2026-10-30) and exist nowhere else — plus the gitignore exception `!src/grounded_vision/data/`.

Current state on a fresh clone at `96e0225`: all subpackages and `pipeline.py` import cleanly; `pytest tests/` reports **16 passed, 1 skipped** (the skip is the real-CUB-312 vocabulary test, which by design requires `$GROUNDED_VISION_CUB_DIR` pointing at a real CUB checkout).

### Other known small issues

- The CRIL-track scripts hard-code output dirs under `/Users/cril/tanmoy/research/data/grounded_vision_runs/` — adjust `OUT`/`RUN` when running elsewhere. (Full train/test feature caches now also exist locally at `/Users/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full/`.)
- `evaluation/interventions.py` annotates `Optional` without importing it — harmless under `from __future__ import annotations`.

---

## Related pages

- [`EXPERIMENT_TRACKS.md`](EXPERIMENT_TRACKS.md) — every script, its inputs/outputs, and where it can run
- [`RESULTS.md`](RESULTS.md) — consolidated, up-to-date quantitative results
- [`ACTIVATION_GROUNDING_WIKI.md`](ACTIVATION_GROUNDING_WIKI.md) — activation-grounding research log
- [`../../grounded_vision/README.md`](../../grounded_vision/README.md) — package README and quickstart
