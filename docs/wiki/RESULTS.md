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

Provenance: run on macOS (Python 3.12, scikit-learn 1.5.1) at repo commit `ba6776d`, via an import shim stubbing the then-uncommitted `grounded_vision.data` module. Fully reproducible: seed 42 is fixed — re-verified 2026-11-02 by rerunning on a second machine (Python 3.11, same scikit-learn 1.5.1): bit-identical outputs, with `selection_accuracy` (0.7033) now recorded in the JSON and a `--out` flag replacing the hardcoded output dir. (Since 2026-11-01 `data/cub.py` is back in the repo and no shim is needed — see [`GROUNDED_VISION_FRAMEWORK.md`](GROUNDED_VISION_FRAMEWORK.md).)

Result file: [`results/clevr_relational_results.json`](../../grounded_vision/results/clevr_relational_results.json).

## 3. Core-pipeline smoke benchmark, synthetic mock CUB (2026-11-01)

Fresh end-to-end run of the six core pipeline stages (`cache_features` → `fit_perception` → `export_predicates` → `train_head` → `evaluate`) with `configs/cub_bound.yaml` on a fresh clone, at commit `96e0225`, on macOS (real DINOv2 ViT-S/14 weights on MPS over the 120+120 synthetic mock images, GT regions). **This is the synthetic smoke path — not a real-data result.** It verifies the full machinery (perception → grounding → certified trees → explanations → audit → interventions) end-to-end; classification numbers are meaningless by construction because the mock labels are not learnable from 120 samples.| Metric group | Values |
| --- | --- |
| Perception | mean part IoU 0.757, PCK 0.943, attribute accuracy 0.858, precision 0.504 / recall 0.488 / F1 0.496 |
| Classification (120 test samples) | accuracy 2.5%, macro-F1 0.021, ECE 0.249 — mock labels, not informative |
| Explanations (120 samples) | **sufficiency rate 100%**, mean sparsity 4.38 ± 0.78 conditions, **NL faithfulness 100%, precision 1.0, recall 1.0, hallucination rate 0%** |
| Interventions (top-5 tree features) | mean prediction flip rate 43.8% |

Scope of the NL numbers: the audit checks the text against the **certified conditions** — NL faithfulness here means *faithful to the model's computation* over predicted concept values, not to the image. If a concept reader misreads the image, the sentence is still faithful to the model and wrong about the bird (the auditor is also lexical, not binding-based — roadmap Requirement 3).

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

**Finding 2 — region geometry (control adjacency) explains most of the v1 crown/breast failure.** With *farthest* controls, crown probes recover to 2.69–4.35 (4/4 pass) and breast to 1.51–2.76 (5/5 pass): the v1 "failure" was largely the anatomically adjacent control box overlapping the same visual content. The *distance-matched* control (same distance as adjacent, but off any part) gives intermediate ratios (crown ~1.3–1.6, breast ~1.4–1.7), decomposing the control effect into a content component (landing on another meaningful part) and a distance component.

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

## 6. Absence-clause observability and perception-training defects (2026-11-04) — NEW

Two data-layer defects found while answering the second external review, both now fixed in code but **predating the deployed `cub_bound_full` checkpoint** (its numbers above stand as measured; reader retraining is queued):

1. **CUB `parts.txt` names parts with spaces** ("left wing"); the parser stored them verbatim while the codebase keys underscores — so `left/right wing/eye/leg` (6 of 15 parts) **never matched their annotations**: always template boxes, visibility 0, no provenance, and their attribute readers never trained. Fixed (name normalization at parse, hermetic regression test).
2. **The train feature cache was built with the train transform's random horizontal flip** while reader training pooled unflipped GT boxes — confirmed by the bimodal cosine check (`scripts/check_train_flip.py`): **40.7%** of 1,000 train samples pool `left_wing` better under the mirrored box (mean cos diff −0.004, bimodal at 0); test-cache control 12.6% (mean +0.094; the floor is genuine wing L/R similarity).

Absence-clause observability over the **full test split** after the parser fix (`scripts/analyze_unobserved_absence.py`; deployed tree + checkpoint, GT regions):

| Metric | GT regions | Predicted regions |
| --- | ---: | ---: |
| Samples / conditions / negated conditions | 5,794 / 61,189 / 40,570 | 5,794 / 61,088 / 40,040 |
| **Absence clauses over unobserved regions** | **7.1%** (2,897) | **3.3%** (1,335) |
| Positive clauses over unobserved regions | 0 | 0 |
| Top unobserved parts (absence clauses) | belly 1,213; breast 806; has_part(belly) 354 | belly 679; has_part(belly) 267; breast 265 |

Reading: after the parser fix, "unobserved" means genuine CUB invisibility (belly/breast on head-on birds) — the earlier 34% estimate from 8 samples was inflated by the name-mismatch bug. Predicted regions show fewer unobserved clauses because the detector proposes boxes liberally — more parts read as observed, but with unverified region quality; GT visibility is the honest reference. Per roadmap Requirement 1, an unobserved part should eventually verbalize as "unobserved", not as absence.

Result files: [`results/unobserved_absence_gt_regions.json`](../../grounded_vision/results/unobserved_absence_gt_regions.json), [`results/unobserved_absence_predicted_regions.json`](../../grounded_vision/results/unobserved_absence_predicted_regions.json), [`results/train_flip_check.json`](../../grounded_vision/results/train_flip_check.json) (scripts: `analyze_unobserved_absence.py`, `check_train_flip.py`).

### 6b. The concept-correctness gate (pre-registered, run 2026-11-04)

The CVPR go/no-go gate, with thresholds written down **before** the rerun. Two post-fix retraining attempts are both reported (defaults, then the documented known-good protocol attr-weight 1.0 / 200 epochs — the lesson from the 312 run); no selection between them. GT regions, full test split (n=5,794; SHAP n=1,000).

| Metric | Pre-fix checkpoint | Post-fix (defaults) | Post-fix (known-good) | Pre-registered bar |
| --- | ---: | ---: | ---: | ---: |
| Species accuracy | 22.68% | 15.83% | **21.11%** | ≥ 22% |
| Perception part IoU / attr acc | — | 0.209 / 71.8% | 0.321 / 76.9% | — |
| Sufficiency / mean reason size | 100% / ~10.7 | 100% / 10.97 | 100% / 10.80 | 100% / ≤ 12 |
| Clause image-correctness (observed regions) | 77.0% | 69.5% | **74.4%** | ≥ 80% |
| Predicted-present clause correctness | 46.1% | — | — | — |
| **P(all clauses correct \| certified reason)** | **7.5%** | 2.5% | **5.2%** | ≥ 60% |
| Absence clauses over unobserved regions | 7.1% | 10.4% | 9.3% | ≤ 8% |
| **SHAP top-k certification failure rate** (matched size, 91–93% overlap) | **73.1%** | 74% | **55.3%** | any nonzero |
| NL faithfulness (binding audit, mode-split) | — | 86.9% | 84.4% | — |

Verdict per the pre-registered bars: **no-go** — accuracy, clause correctness, and reason-level correctness all miss. The striking result is the direction: even after the data-layer fixes and the known-good retraining, only **5.2% of certified sufficient reasons have every clause true of the image**, while 100% of them are verifier-sufficient — the pre-fix 7.5% was flattered by trivially-true absences (the never-observed wings). The SHAP sentence holds post-fix (top-k at matched size fails certification 55.3% of the time despite 93.3% overlap). Also new: the binding audit gives the first non-trivial NL-faithfulness number (84.4% — the old 100% was the token-level audit).

Result files (post-fix = known-good protocol): [`concept_correctness_gt_regions_v2.json`](../../grounded_vision/results/concept_correctness_gt_regions_v2.json), [`shap_certification_gt_regions_v2.json`](../../grounded_vision/results/shap_certification_gt_regions_v2.json), [`unobserved_absence_gt_regions_v2.json`](../../grounded_vision/results/unobserved_absence_gt_regions_v2.json), [`benchmark_results_v2.json`](../../grounded_vision/results/benchmark_results_v2.json); pre-fix versions without the suffix. Scripts: `analyze_concept_correctness.py`, `analyze_shap_certification.py`; config `configs/cub_bound_full_v2.yaml` (eval-transform cache; old run preserved untouched).

### 6c. Corrected headline metrics and the selective-explanations result (2026-11-04, same checkpoint as 6b)

Fourth-round review corrections, all measured on the post-fix known-good checkpoint (GT regions, full test split):

**The 5.2% reason-level number is mostly arithmetic — confirmed.** Clause accuracy 75.0% at a mean 10.8 clauses predicts 0.750^10.8 = **4.48%** under independence; observed **5.21%** (ratio 1.16). The distinctive exhibits are the polarity asymmetry and the selective curve below, not the reason-level rate.

| Metric (post-fix, GT regions) | Value |
| --- | ---: |
| Present-clause precision (raw / class-majority-denoised labels) | **41.7% / 37.9%** |
| Absent-clause accuracy (raw / denoised) vs "always-absent" baseline 86.1% | 91.7% / 94.7% (**lift +5.6 / +8.6 pts**) |
| Base rate of present among certified clauses | 41.7% |
| SHAP top-k certification failure at matched size (n=1,000) | **55.3%** [bootstrap CI95 52.1–58.4], overlap 93.3% |
| SHAP minimal sufficient k (found ≤3× certified for 100%) | mean 11.6 vs certified 10.4 (**excess +1.2**) |
| Accuracy anchors: logreg on same predicted concepts / tree on GT concepts (test) | **32.2%** / 15.6% (train 22.7%) |
| Deployed tree on same readers | 21.1% → the interpretable head costs ~11 pts vs linear |
| NL faithfulness (binding audit, species-name artifact fixed) | **100%** (the 84.4% in §6b was the auditor flagging color words inside CUB species names — `Brown_Booby` — not real failures; fixed + regression test) |

**Selective grounded explanations (the mechanism half):** threshold a certified reason on the minimum reader probability over its present clauses.

| τ | coverage | P(all clauses correct \| kept) | present-clause precision \| kept |
| ---: | ---: | ---: | ---: |
| (none) | 100% | 5.2% | 41.7% |
| 0.6 | 6.0% | 28.5% | 73.8% |
| 0.7 | 1.9% | 43.1% | 94.6% |
| ≥ 0.8 | 1.3% | 46.7% | 100% |

AUROC of min-present-confidence for predicting all-clauses-correct: **0.750** (n=5,793 with reader confidence). Reader confidence cleanly predicts which certified explanations are image-correct — an abstention/flagging mechanism exists, at low coverage.

Caveats recorded: (i) the attr-weight/epoch protocol was chosen after the default failed on the *test-side* gate (both attempts reported; future protocol selection moves to a validation split); (ii) present-clause *recall* is not reported — the per-reason dump does not count GT-positive attributes asserted absent (todo); (iii) SHAP-never-claims-sufficiency positioning and the formal-XAI citation for "rankings are not proofs" belong in the paper text.

Result files: [`selective_explanations_gt_regions_v2.json`](../../grounded_vision/results/selective_explanations_gt_regions_v2.json) (+ `_reasons.jsonl` per-reason dump), [`shap_certification_gt_regions_v2.json`](../../grounded_vision/results/shap_certification_gt_regions_v2.json), [`concept_ceilings_gt_regions_v2.json`](../../grounded_vision/results/concept_ceilings_gt_regions_v2.json). Scripts: `analyze_selective_explanations.py`, `analyze_concept_ceilings.py`.

### 6d. Round-5 controls: the selective curve is length-confounded; the reliability lever does not move (2026-11-04, same checkpoint)

**Confound control (the curve's AUROC is mostly reason length).** AUROC for predicting all-clauses-correct: min-present-confidence **0.750**, but present-clause COUNT alone **0.789**, total clause count 0.364. Within present-clause-count strata, confidence's AUROC falls to 0.704 (n=1), 0.622 (n=2), 0.591 (n=3), 0.734 (n>=4). Zero-present-clause reasons: 1/5,794 (excluded from every tau; no vacuous pass). At the tau=0.8 operating point (n_kept=77): P(all correct) 46.7% [Wilson CI95 34.8-56.5]; present precision 1.000 [95.2-100]; absent-clause accuracy among kept 91.9%. **Composition:** 65.1% of certified clauses are absences (mean 7.2 absent vs 3.6 present per reason). Honest statement: confidence carries weak-to-moderate within-stratum signal; the headline AUROC was inflated by reason length.

**Reliability-ordered minimization (preregistered drop-least-reliable-first): no effect — exactly.** Mean size 10.798 vs 10.798, P(all correct) 5.212% vs 5.212% (n=5,794, verified 100%): greedy backward elimination from the decision path is ORDER-INVARIANT on this tree (path order vs reversed order: 150/150 identical) — the path-reachable prime implicant is unique per sample, so drop-order reordering cannot select a different reason. A reliability-aware choice would need to search beyond path-reachable implicants or trade size for reliability with non-minimal sufficient sets.

**Ceilings completed (label-noise control added):** logreg on GT concepts **21.4%** (vs GT-concept tree 15.6%, predicted-concepts logreg 32.2%, deployed tree 21.1%). Both earlier hypotheses contribute: GT per-image label noise caps even a linear GT head at ~21%, and predicted concepts exceeding GT concepts (~11 pts) confirms reader leakage past the fix.

**Reason-length tax (design rule, closed form):** per-clause accuracy needed for P(all correct)=0.5: k=4 -> 0.841, k=8 -> 0.917, k=12 -> 0.944. With present-clause precision at ~0.42, no ordering trick reaches 50% at k~11; readers must improve or reasons must shrink.

**Nov-16 implication (per the pre-agreed rule):** the mechanism experiment did not lift P(all correct) at full coverage (it cannot move), and the selective curve is length-confounded -> aim for a later venue; spend the time on the reader-quality curve, three-valued predicates, and the LLM arm.

Result files: [`reliability_minimization_gt_regions_v2.json`](../../grounded_vision/results/reliability_minimization_gt_regions_v2.json), [`concept_ceilings_gt_regions_v2.json`](../../grounded_vision/results/concept_ceilings_gt_regions_v2.json). Script: `analyze_reliability_minimization.py` (includes all confound controls).

## 7. Trusted-only reasons and E1 on real CUB (2026-11-06) — NEW

The experiment that decides whether "trusted-only reasons give a guarantee" (review sentence 3) is a *result* or stays a *lemma*: exclusions derived from **independent evidence** (per-concept agreement on a held-out 15% validation sub-split; trust set chosen before any test contact), evaluated on 500 held-out test images. Substrate: eval-transform caches (no flip), `AttributeReader` heads trained per the `fit_perception` protocol on the other 85% of train, CART (depth 12, leaf 4, 200 classes) on predicted 36-concept BOUND vectors; PyXAI exclusion via the Modal job. Scripts: `run_trusted_reasons_cub.py` (build/evaluate) + `run_pyxai_modal.py`; artifacts: [`results/trusted_reasons_cub_summary.json`](../../grounded_vision/results/trusted_reasons_cub_summary.json) + per-arm files.

**The deciding table** (GT = annotated concept vector):

| Row | Coverage | P(all clauses GT-correct \| covered) | Independence prediction | Mean reason size |
| --- | ---: | ---: | ---: | ---: |
| Default reason | 100% | **2.4%** | 3.5% | 10.80 |
| Trusted-only δ=0.60 (33/36 trusted) | 28.8% (82/285) | **1.2%** | 7.5% | 9.95 |
| Trusted-only δ=0.65 (27/36) | 4.1% (19/468) | 0.0% | 14.4% | 8.26 |
| Trusted-only δ=0.70 (21/36) | 0.6% (3/494) | 0.0% | 28.5% | 5.67 |
| Trusted-only δ=0.75 (17/36) | 0.0% (0/500) | — | — | — |

**Verdict, by the pre-agreed rule — held, with the reading corrected (2026-11-06, follow-up review):** the δ=0.60 comparison (1.2% of 82 covered, ~2 events) is **statistically indistinguishable** from the default (Wilson 1.4–4.1% at n=500), so "below the default" is not supported; what *is* supported: coverage collapses with δ (28.8% → 0%) while conditional correctness never exceeds the default, and the crash-bias at δ=0.60 (214/500 queries) means the only useful-coverage row comes from a selected subset. The error-correlation claim is no longer inferred from two events — it is measured directly in §8 (variance of per-reason wrong-clause counts vs binomial). Sentence 3 stays a lemma for the 36-concept/200-class setting; §8 shows where the boundary actually sits. Zero excluded-feature leaks in every arm (the guarantee machinery is sound), and PyXAI's independent audit re-verified native reasons 499/499–500/500 per arm.

**E1** (same run): P(tree(predicted concepts) = tree(GT concepts)) = **2.6%** vs P(all clauses of the default reason GT-correct) = **2.4%** — the certificate event is nearly tight against the output event: the *reason* captures almost everything the output-match condition requires at this scale.

Caveats: pyxai 2.0.1 exclusion-query crashes (`PrimeImplicantCNF` AttributeError) on 214/30/4/0 instances at δ=0.60/0.65/0.70/0.75 — coverage is computed over completed queries and could be biased at δ=0.60; readers trained reader-only on the 85% sub-split (full-cache training may shift absolute numbers, not the qualitative verdict); concept labels are noisy and the vocabulary is 36 concepts; reason size 10.8 at 200 classes makes the all-clauses event intrinsically unlikely — consistent with §6d's reason-length tax.

## 8. Guarantee scaling with reason length (2026-11-06) — NEW

The experiment the follow-up review pointed to: reason length k is what limits the guarantee (P(all clauses correct) ≈ p^k), so sweep the class count K with capped-depth trees (CART depth 12 / leaf 4) and measure where the guarantee becomes useful. Same substrate as §7 (eval-transform caches, readers on the 85% sub-split, per-concept validation reliability + isotonic calibrators on the 15%); **native** trusted-coverage test (exact reachable-leaf DFS — no PyXAI, no crashes) on **all** test images of each class subset; Wilson intervals; direct correlation test (variance of per-reason wrong-clause counts vs the binomial prediction at the reason's own per-concept rates). Script: `run_guarantee_scaling_cub.py`; artifacts: [`results/guarantee_scaling_cub.json`](../../grounded_vision/results/guarantee_scaling_cub.json), [`assets/guarantee_scaling.png`](assets/guarantee_scaling.png).

| K classes | Test n | Tree acc | Mean k | P(all clauses correct) [Wilson 95%] | Present / absent clause precision | Correlation var-ratio |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| 4 | 118 | 60.2% | 3.61 | **25.4%** [18.4, 34.0] | 15.4% / 87.1% | 5.30 |
| 10 | 243 | 44.4% | 5.88 | 11.5% [8.1, 16.1] | 22.7% / 91.4% | 3.06 |
| 20 | 515 | 41.6% | 7.14 | 7.0% [5.1, 9.5] | 32.8% / 91.6% | 1.64 |
| 50 | 1,389 | 30.3% | 8.56 | 3.5% [2.3, 5.3] | 35.4% / 90.1% | 1.03 |
| 100 | 2,864 | 24.5% | 9.67 | 3.0% [1.9, 4.7] | 34.7% / 91.0% | 1.25 |
| 200 | 5,794 | 19.3% | 10.76 | 1.7% [0.9, 3.0] | 32.8% / 89.5% | 0.96 |

**Reading 1 — the scaling law is monotone and measured**: the guarantee is useful exactly where reasons are short (25% at k≈3.6 — the 4-species regime of the activation track, consistent with its 3.79-condition reasons) and collapses to 1.7% at k≈10.8 on all 5,794 test images. The K=200 full-split estimate (1.7% [0.9, 3.0]) supersedes §7's n=500 subsample value and contains it.

**Reading 2 — clause errors are correlated precisely where it matters**: the direct variance test gives var-ratio 5.3 at K=4 and 3.1 at K=10 (strong positive correlation of wrong clauses), converging to binomial (~1.0) at K≥50. So independence *understates* the damage for short reasons — errors cluster — while the p^k arithmetic is roughly right at long ones.

**Reading 3 — neither trust rule rescues the guarantee, anywhere**: per-instance calibrated literal trust (isotonic, per-concept, unobserved→0) covers 25.4% at K=4 but only **7–13%** of covered instances have all kept literals GT-true (kept sets ≈ 22–27 literals); at K≥10 coverage is ≤6.2% and falls to ~0. The global reliability rule covers 25.4% at K=4 (P(kept true | covered) = 10%) and **0% at every K ≥ 10**. Trust-based restriction cannot manufacture a usable guarantee at this vocabulary scale — sentence 3's lemma status is now established crash-free, full-split, under both trust rules.

E1 confirmation (follow-up question): one and the same fitted tree evaluated on predicted vs GT concept vectors (no second tree is trained) — the tightness reading stands.

## 9. Refresh protocol

1. Run the experiment on the machine holding the caches (or via Modal for GPU/PyXAI work).
2. Copy the JSON/PNG artifacts into `grounded_vision/results/` (keep script-defined filenames).
3. Update the matching section here and the "Last verified" date; every table value must trace to a committed file.
