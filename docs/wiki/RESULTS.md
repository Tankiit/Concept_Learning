# Quantitative results — consolidated index

**Last verified:** 2026-11-01 (every number below re-checked against the JSON files in [`grounded_vision/results/`](../../grounded_vision/results/)).

---

## 1. Activation-tree grounding on real CUB (2026-10-27)

> **⚠ invalidated by the tile-pooling bug (found 2026-11-01, external review).** The pooling line in `run_activation_tree_grounding.py` averaged over the wrong axes (`mean(axis=(4,5))` instead of `(3,5)`), so the 16 "tiles" were actually full-width row strips. The numbers below describe a model over row-strip features; every tile box, attached part name, and exemplar crop in the derived artifacts is invalid. The script is fixed; **regeneration on the server (where `train_features.npz` lives) is pending**, after which this section and `results/activation_*.json`, `shap_vs_sufficient_results.json`, `grounding_selected_reasons.json`, the two evidence PNGs, and `NLX_EXAMPLES.md` must be replaced. See [`REVIEW_VALIDATION_ROADMAP.md`](REVIEW_VALIDATION_ROADMAP.md).

Setup: 100 real CUB-200-2011 training images (4 species, 30/30/30/10), frozen DINOv2 feature maps `[384,16,16]` average-pooled to a 4×4 tile grid → 6,144 features; depth-6 CART (Gini, `min_samples_leaf=2`, seed 42); reasons certified by the independent tree verifier.

| Metric | Value | Source |
| --- | ---: | --- |
| In-cache tree accuracy | 95.0% | `activation_tree_grounded_metrics.json` |
| Verified sufficient-reason rate | 100.0% | `activation_tree_grounded_metrics.json` |
| Mean sufficient-reason size | 3.8 conditions | `activation_tree_grounded_metrics.json` |
| TreeSHAP top-|reason| overlap with the certified reason | 88.6% (mean reason size 3.83) | `shap_vs_sufficient_results.json` |

Interpretation (unchanged): TreeSHAP ranks attribution but does not certify sufficiency; the grounded reasons are verifier-sufficient subsets. Details and per-image data: [`ACTIVATION_GROUNDING_WIKI.md`](ACTIVATION_GROUNDING_WIKI.md), [`grounding_selected_reasons.json`](../../grounded_vision/results/grounding_selected_reasons.json), [`shap_vs_sufficient_results.json`](../../grounded_vision/results/shap_vs_sufficient_results.json).

**Cache-only DINO baseline** (same 100 images, thresholded CLS-token tree, exact path certificates): 98.0% training accuracy, mean certificate path length 3.30 — no spatial grounding; see [`cache_only_explainer_summary.md`](cache_only_explainer_summary.md).

## 2. Certified relational reasoning on CLEVR-style scenes (2026-11-01) — NEW

Setup: `scripts/run_clevr_relational.py`, synthetic scenes from the documented CLEVR generative process (seed 42), 4,000 train / 1,000 test scenes, one balanced existential–relational question per scene ("does any red cube stand left of a sphere?"). Features: existential concept literals + relational literals over pairwise spatial facts; tree selected on a 15% validation carve-out (`leaves_64`), refit on full train; native certified sufficient reasons, contrastives, and a leave-one-condition-out necessity audit on 100 samples.

| Metric | Value |
| --- | ---: |
| Test accuracy | 69.8% (validation selection accuracy 70.3%) |
| Mean certified reason size | 4.92 conditions (max 15) |
| Reasons citing ≥ 1 relational literal | 200 / 200 (100%) |
| Mean relational literals per reason | 4.49 |
| Necessity rate (removing any condition breaks sufficiency) | 100% |
| Positive answer rate (test) | 72.6% |

Example (test scene 931, question "gray cylinder front_of cylinder?", answer = prediction = yes): reason = *NOT exists(front_of(cyan sphere, sphere))* ∧ *exists(right_of(yellow cylinder, cylinder))* ∧ *exists(behind(yellow cylinder, sphere))*; contrastive literals flip on `exists(right_of(yellow cylinder, cylinder)) ≤ 0.5` or `exists(behind(red cube, cylinder)) > 0.5`.

Reading: every certified explanation on this track is genuinely **relational** (all reasons cite spatial predicates) and **minimal** (every condition is necessary), but the 69.8% accuracy shows a 64-leaf concept tree does not fully solve existential QA over these scenes — the label is existential over *all* pairs while features encode per-literal existence, so residual approximation error sits in the feature compilation, not the proof layer. The 72.6% positive rate also shows the question generator's 50/50 construction does not survive existential semantics (a sampled-false pair question can still be true via a different matching pair — this is correct label semantics, not a bug).

Provenance: run on macOS (Python 3.12, scikit-learn 1.5.1) at repo commit `ba6776d`, via an import shim stubbing the then-uncommitted `grounded_vision.data` module. Fully reproducible: seed 42 is fixed; adjust the script's `OUT` dir when running off-server. (Since 2026-11-01 `data/cub.py` is back in the repo and no shim is needed — see [`GROUNDED_VISION_FRAMEWORK.md`](GROUNDED_VISION_FRAMEWORK.md).)

Result file: [`results/clevr_relational_results.json`](../../grounded_vision/results/clevr_relational_results.json).

## 3. Core-pipeline smoke benchmark, synthetic mock CUB (2026-11-01)

Fresh end-to-end run of the six core pipeline stages (`cache_features` → `fit_perception` → `export_predicates` → `train_head` → `evaluate`) with `configs/cub_bound.yaml` on a fresh clone, at commit `96e0225`, on macOS (real DINOv2 ViT-S/14 weights on MPS over the 120+120 synthetic mock images, GT regions). **This is the synthetic smoke path — not a real-data result.** It verifies the full machinery (perception → grounding → certified trees → explanations → audit → interventions) end-to-end; classification numbers are meaningless by construction because the mock labels are not learnable from 120 samples.| Metric group | Values |
| --- | --- |
| Perception | mean part IoU 0.757, PCK 0.943, attribute accuracy 0.858, precision 0.504 / recall 0.488 / F1 0.496 |
| Classification (120 test samples) | accuracy 2.5%, macro-F1 0.021, ECE 0.249 — mock labels, not informative |
| Explanations (120 samples) | **sufficiency rate 100%**, mean sparsity 4.38 ± 0.78 conditions, **NL faithfulness 100%, precision 1.0, recall 1.0, hallucination rate 0%** |
| Interventions (top-5 tree features) | mean prediction flip rate 43.8% |

Result file: [`results/cub_bound_mock_benchmark_results.json`](../../grounded_vision/results/cub_bound_mock_benchmark_results.json).

*Superseded variant:* the Modal image also preserved an earlier mock benchmark from the Oct-27 code state (15 evaluated samples: accuracy 73.3%, sufficiency 46.7%, NL faithfulness 86.7%) — replaced by the run above, which reflects current code.

## 4. Predicate grounding milestone — real CUB (2026-11-01) — NEW

First milestone of [`REVIEW_VALIDATION_ROADMAP.md`](REVIEW_VALIDATION_ROADMAP.md): recurring part–attribute predicates evaluated on the **disjoint official CUB test split** (1,500 train / 500 test images), with logistic probes on part-pooled DINOv2 ViT-S/14 features (visible parts only — unobserved parts excluded), and perturbation specificity: blur / gray-mask of the predicate's own part box vs the **same edit geometry placed on a control part**, plus stability under unrelated-part edits. 16 of 19 in-scope predicates were recurring (≥30 train / ≥10 test positives); run on macOS/MPS against the local CUB copy.

| Metric | Value |
| --- | --- |
| Predicate–annotation agreement | mean accuracy **82.8%** (range 71.9–94.9%), AUROC 0.767–0.979 |
| Perturbation specificity (mean target-vs-control \|ΔP\| ratio) | blur **1.99×**, mask **2.49×** |
| Predicates passing (acc ≥ 0.70 AND specificity ≥ 1.5× under both edits) | **7 / 16** |

The split is by part, and it is the headline:

| Part family | Specificity range | Verdict |
| --- | ---: | --- |
| beak (bill shape, bill color) | 2.77–4.25× | **grounded** — all 4 predicates pass; masking the bill flips `bill_color=black` predictions on 45% of images vs 6.9% ΔP under control |
| belly (colors) | 2.72–5.23× | **grounded** — all 3 pass (`belly_color=yellow`: mask 5.23×) |
| breast (colors, pattern) | 1.13–1.69× | agreement high (up to 92.8%) but specificity below threshold under blur — edits bleed into the adjacent belly/throat area |
| crown (colors) | 1.13–1.45× | weakest — small box adjacent to forehead/nape; control edits on the neighboring head region move it nearly as much |

Reading, per the review's decision rule: the beak and belly predicates are **reliable and specific** — certified part–attribute explanations over them are now backed by measured grounding, and expansion to relational explanations can proceed on that basis. Crown and breast predicates are reliable but **not yet specific** — the next work there is region geometry (smaller/steered boxes, distance-matched non-adjacent controls), i.e. perception and grounding, before stronger certification claims. One design caveat cuts in their favor: the control region was the *next scope part in fixed order*, which for crown is the anatomically adjacent beak — adjacent controls inflate the control ΔP, so crown/breast specificity is, if anything, underestimated.

Caveats: probes are milestone instruments (logistic regression on pooled features), not the deployed `AttributeReader` heads — this milestone validates the feature grounding layer those readers depend on; masking/blur earns sensitivity claims, not semantic-correction claims.

Result file: [`results/predicate_grounding_milestone.json`](../../grounded_vision/results/predicate_grounding_milestone.json) (script: `scripts/run_predicate_grounding_milestone.py`, Modal launcher `scripts/run_milestone_modal.py`).

## 5. Pending results — produced on the dev server, not yet in this repo

The reviewer-analysis track (Track C in [`EXPERIMENT_TRACKS.md`](EXPERIMENT_TRACKS.md)) runs only where the CRIL caches exist. After each server run, copy these into `grounded_vision/results/` and add a section above:

| Server artifact (`RUNS = /Users/cril/tanmoy/research/data/grounded_vision_runs`) | Producing script |
| --- | --- |
| `{RUNS}/cub_bound_full/results/analysis_{gt,predicted}_regions.json` | `analyze_full_run.py` |
| `{RUNS}/cub_bound_312/results/ceiling_312.json` | `run_312_ceiling.py` |
| `{RUNS}/cub_bound_full/tree_methods_comparison.json` (+ `rf_pyxai_payload.json`) | `compare_tree_methods.py` |
| `{RUNS}/celeba_concepts/tree_methods_comparison.json` | `compare_tree_methods_binary.py` |
| `{RUNS}/tree_quality_report.json` | `analyze_tree_quality.py` |
| `{RUNS}/cub_bound_full/depth_sweep/depth_sweep_summary.json` | `sweep_tree_depth.py` |
| `{RUNS}/cub_bound_full/tree_search/tree_search_summary.json`, `validated_selection.json` | `search_explanation_trees.py`, `rerun_tree_search_validated.py` |
| `{RUNS}/cub_bound_full/pyxai_export_n<N>.json` (+ `*_pyxai_results.json` after the Modal audit) | `export_pyxai_comparison.py`, `run_pyxai_modal.py` |
| `{RUNS}/cub_bound_full/exemplar_leaves_report.json` | `run_exemplar_leaves.py` |
| `{RUNS}/cub_bound_full/prototype_tree/prototype_tree_summary.json` | `run_prototype_tree.py` |
| `{RUNS}/matryoshka_sae/msae_results.json` | `run_matryoshka_sae.py` |
| Core-pipeline benchmark for any config: `{cache_dir}/results/benchmark_results.json` (the mock-CUB variant is already in the repo — section 3; the real-data variants are not) | `evaluate.py` |

## 6. Refresh protocol

1. Run the experiment on the machine holding the caches (or via Modal for GPU/PyXAI work).
2. Copy the JSON/PNG artifacts into `grounded_vision/results/` (keep script-defined filenames).
3. Update the matching section here and the "Last verified" date; every table value must trace to a committed file.
