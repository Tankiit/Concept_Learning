# Experiment tracks & script catalog

All 31 scripts live in [`grounded_vision/scripts/`](../../grounded_vision/scripts). Paths below abbreviate `RUNS = /Users/cril/tanmoy/research/data/grounded_vision_runs` and `CUB = /Users/cril/tanmoy/research/data/CUB_200_2011/CUB_200_2011` — the development server where the real-data caches live. Full train/test feature caches also exist locally at `/Users/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full/` (5,994 + 5,794 images, built 2026-11-02). See "Where each track can run" per section and [`GROUNDED_VISION_FRAMEWORK.md`](GROUNDED_VISION_FRAMEWORK.md) for the resolved `data/`-package history.

---

## Track A — Core config-driven pipeline

The six canonical stages, driven by a YAML config (`--config configs/cub_bound.yaml` by default). With the synthetic-capable configs (`cub_flat`, `cub_bound`, `cub_relational`) these run end-to-end on a fresh clone without real CUB data (`use_mock_if_missing: true` → synthetic dataset + mock dataset loading; the backbone still uses real DINOv2 weights when available). The full `data/` package has been committed since `96e0225` (2026-11-01).

| Script | What it does | Writes |
| --- | --- | --- |
| `cache_features.py` | Extract frozen DINOv2 CLS tokens + `[384,16,16]` feature maps | `{cache_dir}/{train,test}_features.npz` |
| `fit_perception.py` | Train PartDetector + AttributeReader + prototype bank (needs the train cache; no mock) | `{cache_dir}/perception_checkpoint.pt`, `perception_train_metrics.json` |
| `train_head.py` | Compile predicates → fit `CertifiableTreeHead` | `{cache_dir}/tree_head.pkl` |
| `export_predicates.py` | Export per-sample `FactDatabase`s | `{cache_dir}/{split}_predicates.jsonl` |
| `explain.py` | Certified sufficient + contrastive explanation for one sample (`--sample-idx`), with evidence and NL audit | stdout only |
| `evaluate.py` | Full benchmark: perception IoU/F1, tree accuracy/ECE, explanation sufficiency/sparsity/faithfulness, interventions | `{cache_dir}/results/benchmark_results.json` |

Note `evaluate.py` writes under the config's `cache_dir`, **not** the repo `results/` directory.

## Track B — Activation grounding track (real CUB cache)

**Primary script (official train/test split, 2026-11-02):** `run_activation_track_traintest.py` — caches both splits, fits the depth-6 activation tree on train only, evaluates held-out accuracy / verified rate / reason size / grounding selection / TreeSHAP overlap on test, and renders test-image evidence tiles + train→test exemplar retrieval. Writes `activation_track_traintest_results{,_c4}.json` and the `activation_traintest_*.png` assets. **Runnable locally** against `/Users/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full`.

Historical one-off chain over the 100-image in-cache protocol (paths patched from CRIL; pooling fixed 2026-11-01/02):

| Script | What it does | Writes |
| --- | --- | --- |
| `run_activation_tree_grounding.py` | Depth-6 CART on 4×4-pooled activations (6,144 features); certified reasons with spatial tile grounding | `activation_tree_grounded_explanations.jsonl`, `..._examples.png`, `..._metrics.json` |
| `select_grounded_reasons.py` | Enumerate all sufficient subsets per decision path; select the one covering most visible CUB part points | `grounding_selected_reasons.json` |
| `compare_shap_grounding.py` | TreeSHAP top-|reason| attribution vs verifier-sufficient reasons | `shap_vs_sufficient_results.json` |
| `make_activation_nlx.py` | Template NLX over certified activation reasons | `activation_grounded_nlx.jsonl` |
| `ground_tiles_to_cub_parts.py` | Attach validated CUB part-point semantics to evidence tiles | `activation_part_grounded_nlx.jsonl` |
| `add_exemplar_grounding.py` | Nearest-neighbour exemplar crops per certified tile | `activation_exemplar_grounded_nlx.jsonl`, `activation_exemplar_examples.png` |
| `explain_cached_dino.py` | Cache-only baseline: thresholded CLS-token tree with exact path certificates (see [`cache_only_explainer_summary.md`](cache_only_explainer_summary.md)) | `cache_only_explanations.jsonl`, `cache_only_explainer_summary.md` |

## Track C — Reviewer-analysis track (server-only: needs CRIL caches + checkpoints)

Config `configs/cub_bound_full.yaml` unless noted; cache `{RUNS}/cub_bound_full`.

| Script | What it does | Writes |
| --- | --- | --- |
| `analyze_full_run.py` | Concept ceiling (tree/logreg/MLP on predicted vs GT concepts), reason compression + necessity, per-attribute AUROC/F1 vs tree splits, explanation-relative flips | `{cache_dir}/results/analysis_{regions}_regions.json` |
| `run_312_ceiling.py` | Concept ceiling with all 312 GT attributes + per-attribute logistic probes on pooled part embeddings (`cub_bound_312.yaml`) | `{cache_dir}/results/ceiling_312.json` |
| `compare_tree_methods.py` | GOSDT (optimal sparse trees) / RuleFit / RandomForest vs CART, scored accuracy-vs-certified-proof-size; exports an RF payload for PyXAI | `{RUNS}/cub_bound_full/tree_methods_comparison.json`, `rf_pyxai_payload.json` |
| `compare_tree_methods_binary.py` | Same comparison on binary CelebA `Male` (`celeba_concepts.yaml`) | `{RUNS}/celeba_concepts/tree_methods_comparison.json` |
| `analyze_tree_quality.py` | Structural quality of saved tree pickles: depth, leaf purity, impure (vacuous-certificate) leaves, memorization masses, feature usage, unused concepts; cross-tree comparisons | `{RUNS}/tree_quality_report.json` |
| `sweep_tree_depth.py` | CART accuracy/reason-size trade-off at depths 2–12 | `{cache_dir}/depth_sweep/depth_sweep_summary.json` + per-depth PyXAI payloads |
| `search_explanation_trees.py` | Asymmetric-tree search (ccp/leaves/msl families) for the accuracy-vs-reason-size Pareto frontier | `{cache_dir}/tree_search/tree_search_summary.json` + `frontier_*.json` |
| `rerun_tree_search_validated.py` | Methodological fix of the search: candidate selection on a 15% validation carve-out, refit on full train, single test report; **overwrites the deployed `tree_head.pkl`** | `{cache_dir}/tree_search/validated_selection.json` |
| `export_pyxai_comparison.py` | Export deployed concept tree + 200 test instances with native greedy reasons for the PyXAI minimum-cardinality audit | `{cache_dir}/pyxai_export_n<N>.json` |
| `run_exemplar_leaves.py` | Case-based evidence: nearest same-leaf training exemplars per test instance | `{cache_dir}/exemplar_leaves_report.json` |
| `run_prototype_tree.py` | Prototype-similarity threshold tree ("cosine to prototype_k ≥ τ") with certified reasons | `{cache_dir}/prototype_tree/prototype_tree_summary.json` + ~9 candidate payloads |
| `run_matryoshka_sae.py` | TopK Matryoshka SAE (levels 32/128/512) on DINO tiles; CART per level, certified reasons, part grounding, latent-attribute correlation | `{RUNS}/matryoshka_sae/msae_results.json`, `msae_fine_pyxai.json` |

## Track D — Relational track (CLEVR) — runs anywhere

`run_clevr_relational.py` — certified relational QA over CLEVR-style scene graphs ("is there a red cube left of a sphere?"). Scene graphs only (oracle grounding, like `--regions gt` on CUB); no images or backbone needed. Fully synthetic by default (documented CLEVR generative process, seed 42); `--scenes-json` accepts real CLEVR `scenes.json`. Pipeline: scene → `FactDatabase` (object entities, `has_attribute`, pairwise spatial facts) → existential/relational features → validation-selected tree → native certified reasons + contrastives + necessity audit.

Only blocker off-server: the hard-coded `OUT` directory. **Result:** [`results/clevr_relational_results.json`](../../grounded_vision/results/clevr_relational_results.json) (see [`RESULTS.md`](RESULTS.md)). Runs on a fresh clone.

## Track F — Validation milestone (review requirement 2) — runs locally

`run_predicate_grounding_milestone.py` (+ Modal launcher `run_milestone_modal.py`) — the first milestone of [`REVIEW_VALIDATION_ROADMAP.md`](REVIEW_VALIDATION_ROADMAP.md): recurring part–attribute predicates on real CUB, official disjoint test split, logistic probes on part-pooled DINOv2 features (only where the part is *visible* — unknown excluded), and perturbation specificity: blur/gray-mask the predicate's own part box vs the same edit geometry placed on a control part, plus stability under unrelated-part edits. Writes `results/predicate_grounding_milestone.json`. Requires a real CUB checkout (`--data-dir`; local copy at `/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011`, also on the `grounded-vision-cub-data` Modal volume — `attributes.txt` restored there 2026-11-01).

## Track E — Cloud jobs (Modal)

| Script | What it does | Writes |
| --- | --- | --- |
| `run_grounded_vision_modal.py` | Whole 6-stage pipeline on CUB (T4 GPU) against Modal volumes | `/results/cub_bound_real_full` on the `grounded-vision-cub-results` volume |
| `run_pyxai_modal.py` | PyXAI minimum-cardinality sufficient reasons on an exported payload (`--data <payload.json>`, produced by Track C/D scripts); supports majority-vote Random-Forest ensemble payloads | `<payload>_pyxai_results.json` next to the input |

---

## Where each track can run (as of 2026-11-01)

| Track | Fresh clone / laptop | Notes |
| --- | --- | --- |
| A — core pipeline | ✅ (mock configs) | `pytest tests/`: 16 passed, 1 skipped. Mock-config runs work off-server; the real-data configs point at CRIL absolute paths. |
| B — activation track | ✅ (local caches) | Full train/test caches at `/Users/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full` since 2026-11-02; `run_activation_track_traintest.py` runs locally. Regenerated results are in `grounded_vision/results/`. |
| C — reviewer analysis | ✗ | Requires CRIL feature caches + fitted checkpoints. Run on the server, then copy the JSONs into `grounded_vision/results/` (see [`RESULTS.md`](RESULTS.md) "pending" list). |
| D — CLEVR relational | ✅ | Synthetic; adjust `OUT`. |
| E — Modal | ✗ (needs `modal` + payloads/local data dir) | `run_grounded_vision_modal.py` also expects the CRIL local dir as its `add_local_dir` source. |
