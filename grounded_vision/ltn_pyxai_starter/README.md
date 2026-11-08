# LTNtorch and PyXAI visual logic pilot

**Start with Boolean predicates and PyXAI:** [new walkthrough](boolean_cbm_walkthrough.ipynb) and [guide](boolean_cbm/README.md). Includes an exact small-rule bridge, a trainable tree head, crop-first readers, and independent reason checks.

**New: the difflogic/CVPR implementation is in [cvpr_logic/README.md](cvpr_logic/README.md).**
It includes 16-operation logic layers, a cached-feature adapter, complete-circuit
SAT explanations, checkpoint inference, tests and an optional upstream CUDA backend.
The instructions below describe the original two-gate pilot, retained as a baseline.

This is a runnable integration test for **a learned logic head that is the actual predictor**, with a visual neural reader, exact Boolean export, and calibrated uncertainty over its premises. It is not distillation and is not a generic LTN-to-tree converter.

## Run

Tested on Linux, Python 3.12, PyTorch 2.8.0+cpu, LTNtorch 1.0.2, PyXAI 2.0.1. Also verified on macOS 15 arm64 with the same pinned versions (all three test suites and short training smokes pass; see the two VALIDATION.md files). PyXAI has several transitive dependencies; allow its installer to resolve them. GUI dependencies are unnecessary. For a CPU-only environment install PyTorch from its CPU index before the requirements.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest test_logic.py
python pilot.py --objective ltn --out results/ltn_seed7
python pilot.py --objective bce --out results/bce_seed7
```

Use Python without `-O`: the pilot deliberately uses assertions to fail on verification mismatches. Some PyXAI dependencies inspect command-line options and print warnings about this script's arguments; these warnings do not affect the saved outputs.

## What is implemented

1. Synthetic RGB images contain three independently sampled attributes in fixed regions. Random region occlusion hides evidence but retains the underlying concept label. This is a controlled plumbing test, not CUB or a realistic image-edit experiment.
2. A small CNN produces three concept scores. A custom two-gate head learns AND versus OR at each node in the fixed topology `g_outer(c0, g_inner(c1,c2))`. There are only four possible discrete formulas. Concept identities and topology are supplied, not discovered.
3. LTNtorch constructs a satisfaction loss for supervised concept and task predicates, using universal p-mean-error aggregation. Fuzzy AND/OR operations provide gradients to gate choices and the reader. The comparison arm uses BCE on the same four supervised outputs and the same architecture. Neither arm is supplied the target AND/OR choices. No co-occurrence assumptions enter the proof theory.
4. Checkpoint selection uses tuning data. The deployed head fixes each gate to its largest-weight operator and thresholds reader outputs at 0.5. The soft/hard prediction difference is measured; no equality to the soft training graph is assumed.
5. All eight Boolean input assignments are evaluated by the deployed head. A decision tree is built directly from that complete function by Shannon expansion. It is **not fitted to sampled teacher outputs**. Its truth table must match exactly before any explanation is accepted.
6. PyXAI enumerates the subset-minimal sufficient reasons for every Boolean assignment. A separate exhaustive implementation checks sufficiency, subset minimality, and enumeration completeness. The ordinary policy selects the shortest reason with deterministic ties.
7. A held-out calibration split constructs a joint concept-vector prediction set with score `max_j(1-q_j if c_j=1 else q_j)`. This maximum handles simultaneous concept coverage without an independence assumption. The finite-sample quantile uses `ceil((n+1)*(1-alpha))`.
8. The robust policy issues a sufficient reason only if every concept vector in the nonempty prediction set satisfies it. Unknown concepts therefore range over both Boolean values. Candidate reasons come from PyXAI; robust filtering is our code.
9. Per-image records expose ordinary and robust explanations, semantic correctness, acceptance, reason length, target, hard/soft prediction, visibility, concept scores, and joint-set coverage. Same-image, same-length effects are reported separately from changes due to abstention.

## What the guarantee means

Assume the learned model, score, and alpha are fixed before independent, exchangeable calibration/test examples. If the joint concept set covers the true annotated concept vector with marginal probability at least `1-alpha`, an issued explanation can contain a false literal only when that set misses the true vector. Thus the marginal event `issued AND false-premise` has probability at most alpha. This is not accepted-set conditional error, per-image certainty, or task-label correctness. Calibration is marginal over calibration and test draws, not a high-probability guarantee conditional on this particular calibration sample.

The exact PyXAI guarantee concerns the deployed Boolean head. Grounding means agreement with the supplied concept annotation in this pilot. It does not establish causal use of the intended image region, or rule out a nuisance-based reader. Neural localization and invariance tests belong in the real-data study.

## Outputs and limitations

Each run saves `metrics.json`, `per_example.jsonl`, `training.json`, `provenance.json`, `model.pt`, `calibration.npz`, `truth_table.csv`, and `sample_images.png`. The delivered run folders are a single-seed synthetic feasibility test. `PILOT_RESULTS.md` interprets them. `RESEARCH_PLAN.md` maps the remaining work to IJCAI and ICML.

The exhaustive exporter scales as `2**m`; this prototype fixes `m=3`. Do not expand it to hundreds of CUB concepts. Use a hard branching architecture with a direct export, or a circuit representation and a suitable SAT-based explainer. LTNtorch itself is not a structure learner, and arbitrary fuzzy formulas cannot generally be compiled to equivalent Boolean trees by thresholding.

No claim that LTN training outperforms BCE is supported by this run. No strong confidence/learned-selector baseline, statistical replication, real-data adapter, direct neural-backbone robustness proof, arbitrary logic-structure search, or general export algorithm is implemented.

## Real-data adapter contract

Replace the generator/reader only after recording stable image IDs and a fixed concept vocabulary. For cached features, fit a neural reader from features to concept scores. Keep all rows from an original image in one split. Required fields are `id`, `features` or image path, `concept_labels`, `concept_observed_mask`, `task_label`, `split`, and optional part locations/visibility/nuisance metadata. Unknown annotations must not become negative labels. Train/tune before calibration; any change to readers, thresholds, gate choices, vocabulary, or scoring policy requires a fresh valid calibration step.

Current code uses fully known synthetic latent labels even for occluded objects. This differs from CUB's missing/uncertain annotations: the real-data protocol must state the observable target and handle missing labels explicitly before applying a joint guarantee.

## Sources and provenance

- [LTNtorch](https://github.com/tommasocarraro/LTNtorch): inspected commit `d1bd98169cc2121f8cdd25ff99901e4589923c95`; installed PyPI 1.0.2.
- [PyXAI](https://github.com/crillab/pyxai): inspected commit `2eb18994832e67b7ce082e357f18b5f2d12631a6`; installed PyPI 2.0.1.
- [LTN predicate grounding](https://logictensornetworks.github.io/LTNtorch/grounding.html).
- [PyXAI Boolean representation](https://www.cril.univ-artois.fr/pyxai/documentation/explainer/concepts/).
- [COCOCO](https://arxiv.org/abs/2605.18202): close prior work for conformal neuro-symbolic models; see the research plan.

No upstream repository was modified. Original integration code is provided for research use; dependencies retain their own licenses.
