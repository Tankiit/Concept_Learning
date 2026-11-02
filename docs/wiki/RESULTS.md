# Quantitative results — consolidated index

**Last verified:** 2026-11-02 (every number below re-checked against the JSON files in [`grounded_vision/results/`](../../grounded_vision/results/)).

---

## 1. Activation-tree grounding on real CUB (regenerated 2026-11-02)

> **Protocol upgrade.** The original track (2026-10-27) fit and evaluated on the same ~100 training images, and its tile pooling averaged over the wrong axes (found in external review, 2026-11-01 — the "tiles" were row strips). Everything below is regenerated from a fresh local cache (`cache_features.py --split all` over the full official split) with the corrected pooling (`mean` over within-tile axes) and the tree fit on TRAIN only, evaluated on held-out TEST.

Setup: frozen DINOv2 feature maps `[384,16,16]` → 4×4 tile grid × 384 channels (6,144 features, corrected pooling); depth-6 CART (Gini, `min_samples_leaf=2`, seed 42) on train; reasons certified on test by the independent verifier. Script: `scripts/run_activation_track_traintest.py`.

**Official-split evaluation (primary result):**

| Metric | 4-species subset (120 train / 118 test) | Full CUB-200 (5,994 train / 5,794 test) |
| --- | ---: | ---: |
| Train accuracy | 95.0% | 8.3% |
| **Test accuracy (held-out)** | **56.8%** | **4.3%** |
| Verified sufficient-reason rate on test | 100% (118/118) | 100% (500/500 sampled) |
| Mean certified reason size on test | 3.79 conditions | 5.98 conditions |
| Grounding-selected reason (size / part score) | 3.90 / 3.02 | 5.98 / 7.97 |
| TreeSHAP top-|reason| overlap on test | 87.9% | 95.1% |

Reading: the historical "95.0% accuracy" was **in-cache** (train) accuracy; under the official split the same protocol generalizes at 56.8% on the 4-species subset. The certificate layer itself is scale-stable — 100% of held-out explanations remain verifier-sufficient with ~4–6-condition reasons and part-grounded tiles at both scales — while the depth-6 activation tree is plainly not a competitive 200-class head (4.3% vs 0.5% chance). That separation (certificates hold, classifier accuracy is the binding constraint) is the track's central honest finding, and the motivation for the concept-tree and tree-method-comparison tracks.

**Historical in-cache protocol, rerun with fixed pooling** (same 100 images as 2026-10-27, identified by sample-id set): accuracy 95.0%, verified 100%, mean reason 3.69; TreeSHAP overlap 85.3%. Kept in `activation_tree_grounded_metrics.json`, `grounding_selected_reasons.json`, `shap_vs_sufficient_results.json` for continuity; all three numbers are train-side by construction.

**Cache-only DINO baseline** (thresholded CLS-token tree, exact path certificates, same 100 images): 95.0% training accuracy, mean certificate path 3.60 conditions — no spatial grounding; see [`cache_only_explainer_summary.md`](cache_only_explainer_summary.md).

Provenance: caches rebuilt locally from `/Users/tanmoy/research/data/CUB_200_2011` (the train transform includes a random horizontal flip, so features are a fresh draw of the same protocol, not bit-identical to the 2026-10-27 cache); `pyxai_detected` is `false` in the rerun metrics because the PyXAI package is not installed locally — certification used the native verifier throughout, as before. Evidence tiles: [`assets/activation_traintest_examples.png`](assets/activation_traintest_examples.png) (test images), [`assets/activation_traintest_exemplars.png`](assets/activation_traintest_exemplars.png) (test→train exemplar retrieval); per-sample data in `activation_track_traintest_results{,_c4}.json`.

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

### Milestone v2 (2026-11-02): deployed readers, richer controls, uncertainty estimates

Script: `scripts/run_predicate_grounding_milestone_v2.py` — same 16 predicates, split, and images as v1, with three upgrades: (i) the **deployed `AttributeReader` heads** trained with the `fit_perception` protocol (per-region linear heads, BCE-with-logits with pos-weight, Adam, 40 epochs, seed 42) evaluated alongside the v1 logistic probes — both reading features pooled by the framework's own `pool_region_features`; (ii) **three control families** — *adjacent* (v1 rule; mean center distance 0.17), *farthest* visible scope part (0.27), and *distance-matched* (same 0.17 distance as adjacent, direction chosen off any part); (iii) **per-image effect records** (30,264 rows, every p0 and signed ΔP per condition) and bootstrap 95% CIs (1,000 resamples) on every specificity ratio. Files: [`predicate_grounding_milestone_v2.json`](../../grounded_vision/results/predicate_grounding_milestone_v2.json), [`predicate_grounding_milestone_v2_effects.jsonl`](../../grounded_vision/results/predicate_grounding_milestone_v2_effects.jsonl).

Mean specificity (target-vs-control |ΔP| ratio) by part family, control family, and instrument:

| Family | Probe: adjacent → farthest → dist-matched | Reader: adjacent → farthest → dist-matched |
| --- | --- | --- |
| beak | 2.85 → 4.42 → 2.64 (passes 4/4 \| 4/4 \| 4/4) | 2.32 → 3.23 → 2.64 (2/4 \| 2/4 \| 2/4) |
| belly | 3.11 → 3.13 → 4.64 (3/3 \| 3/3 \| 3/3) | 2.76 → 2.79 → 3.95 (2/3 \| 2/3 \| 2/3) |
| breast | 1.32 → **2.20** → 1.56 (0/5 \| **5/5** \| 2/5) | 1.37 → 1.87 → 1.48 (0/5 \| 2/5 \| 0/5) |
| crown | 1.25 → **3.34** → 1.41 (0/4 \| **4/4** \| 0/4) | 1.34 → 3.57 → 1.49 (0/4 \| 1/4 \| 1/4) |

**Finding 1 — the separation persists under the deployed readers.** The part-family pattern is unchanged (beak/belly 2.3–4×; crown/breast ~1.3–1.5 under adjacent controls), with tight CIs (e.g. crown `crown_color=black` blur, farthest control: 3.60, CI95 [3.16, 4.12]).

**Finding 2 — region geometry (control adjacency) explains most of the v1 crown/breast failure.** With *farthest* controls, crown probes recover to 2.70–4.35 (4/4 pass) and breast to 1.5–2.7 (5/5 pass): the v1 "failure" was largely the anatomically adjacent control box overlapping the same visual content. The *distance-matched* control (same distance as adjacent, but off any part) gives intermediate ratios (crown ~1.3–1.6, breast ~1.4–1.7), decomposing the control effect into a content component (landing on another meaningful part) and a distance component.

**Finding 3 — the residual bottleneck moves to the reader heads.** Deployed readers agree with annotations at 0.691 mean accuracy (AUROC 0.806) vs 0.824 (0.862) for the balanced probes, and several reader predicates miss the 0.70 accuracy bar despite strong specificity (e.g. crown farthest specificity 3.57 with only 1/4 passing). Per the review's decision rule, the next work is reader training (perception), not the proof layer.

Caveats: masking/blur still establish sensitivity, not semantic correction; reader heads were trained reader-only on the 1,500-image milestone subset (the full-cache `fit_perception` run may close part of the agreement gap).

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
