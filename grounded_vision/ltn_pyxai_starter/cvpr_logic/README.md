# CVPR visual logic starter

A trainable visual concept reader, differentiable logic-gate head, hard deployment,
joint concept calibration, and exact SAT explanations. This adds a general small
logic circuit to the original two-gate LTN/PyXAI pilot.

## Installation and quick start

Run these commands from the extracted **ltn_pyxai_starter** directory. Tested with
Linux, Python 3.12 and CPU PyTorch 2.8.0. No dataset download or GPU is required for
the synthetic demo. Dependencies for PyXAI can take time to install.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest cvpr_logic.test_core
python -m cvpr_logic.train --objective bce --pyxai-check --out results/my_bce
python -m cvpr_logic.train --objective ltn --pyxai-check --out results/my_ltn
```

Use `--ste` for hard concept inputs in the training forward pass with an identity
straight-through derivative. Otherwise training uses continuous concept scores.
Both variants deploy thresholded concepts and argmax gate choices. Discretization
may change accuracy; the saved soft/hard disagreement measures this. Neither
surrogate derivatives nor LTN satisfaction create a logical guarantee.

The default reference backend uses the **unchanged upstream difflogic gate
functions**, vendored under MIT in `vendor/`. Our wrapper adds checkpointed wiring
and avoids the package's unconditional CUDA-extension import. It implements the
same 16-gate relaxation; it is not a benchmark of upstream CUDA performance.
The upstream commit is pinned and its license is included.

## Optional upstream difflogic backend

On a CUDA machine, install a CUDA-enabled PyTorch 2.8.0 build matching your CUDA
Toolkit; `nvcc`, a C++ compiler, git and build tools are required. Do not install
the CPU wheel above for this path. Then:

```bash
python -m pip install -r requirements.txt
python -m pip install setuptools wheel ninja
python -m pip install --no-build-isolation -r cvpr_logic/requirements-cuda.txt
python -m cvpr_logic.train --backend upstream --device cuda --out results/my_cuda
```

The CUDA backend adapter is supplied but **was not executed in this CPU-only
environment**. No claim of tested CUDA-toolchain compatibility is made. Calling
`.eval()` alone does not binarize floating inputs; this starter explicitly uses
`q >= 0.5` for deployment. Upstream input wiring and CUDA backward adjacency are
restored alongside the learned gate weights when loading checkpoints.

## Use your cached CUB/backbone features

Supply one NPZ, loaded with `allow_pickle=False`. Features can come from your
existing frozen DINO reader pipeline; this starter does not download DINO, train
a part detector, or fabricate CUB annotations. Feature preprocessing is fitted
only on the train split. All descendants of one original image must share a split.

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `x` | float32 `[N,D]` or `[N,3,H,W]` | Cached features or consistently preprocessed RGB tensors |
| `c` | `[N,M]` | Binary concept labels; unknown entries may be NaN when unobserved |
| `observed` | bool `[N,M]` | Whether each label is available, not a visibility-to-negative conversion |
| `y` | int64 `[N]` | Class IDs 0 through K-1, every class represented in training |
| `ids` | Unicode `[N]` | Unique example IDs |
| `groups` | Unicode `[N]` | Original-image IDs shared by perturbations |
| `split` | Unicode `[N]` | `train`, `tune`, `calibration`, or `test` |
| `concept_names` | Unicode `[M]` | Fixed, unique predicate names |

```python
import numpy as np
# These arrays must come from your own verified dataset loader.
np.savez_compressed('cub_features.npz',
    x=features.astype('float32'), c=concept_labels,
    observed=concept_observed_mask.astype(bool), y=class_ids.astype('int64'),
    ids=np.asarray(image_ids, dtype=str), groups=np.asarray(original_image_ids, dtype=str),
    split=np.asarray(split_names, dtype=str), concept_names=np.asarray(names, dtype=str))
```

```bash
python -m cvpr_logic.train --data cub_features.npz --width 400 --depth 2 --max-length 4 --out results/cub_logic
```

Width must be a multiple of the number of classes and at least half the number
of concepts; 400 is an illustrative valid width for 200 classes, not a tuned
recommendation. Large multiclass models may exhaust solver/search budgets.
Do not enable `--pyxai-check` above ten concepts: it enumerates the entire input
cube. General explanation uses the compact gate CNF instead.

Training/tuning permit missing concept labels via the mask. **Calibration and
test require complete concept annotations for the declared vocabulary.** The
implemented guarantee targets the complete vector; a mask over a partially
labelled vector is a different target. For CUB, restrict the target/vocabulary or
collect annotations before using this calibration. Visibility and unknownness are
not negative labels. Synthetic occluded attributes remain fully labelled latent
properties and do not solve CUB's missing-annotation problem.

To reuse an already trained, frozen predicate reader, set `x` to its probability
matrix `[N,M]` and add `--precomputed-concepts`. This trains only the logic head;
it does not standardize probabilities or refit the reader. The frozen reader must
have been trained/tuned without calibration or test labels. The inference input
must then also be its probability matrix. This is the preferred first controlled
comparison with existing CUB predicates.

For a functional cached-feature example:

```bash
python -m cvpr_logic.data --out demo_features.npz
python -m cvpr_logic.train --data demo_features.npz --epochs 20 --out results/features
python -m cvpr_logic.infer --run results/features --data demo_features.npz --out predictions.jsonl
```

The inference command needs only `x` and optional `ids`, and never reads concept or
class labels. The command above is an interface smoke test on the full synthetic
archive, not a held-out performance report. Normal training evaluation uses only
its test rows. With complete annotations in every split, `--oracle-concepts`
bypasses the reader to diagnose vocabulary/head capacity. Oracle runs cannot be
used with the deployment inference command.

## Exactness and uncertainty

The hard circuit consumes concept bits, applies selected gates, sums equal groups
of final bits, and selects the lowest class index on a tie. Each gate has a local
CNF truth-table encoding. A cardinality constraint encodes each rival class beating
the target, including ties. A reason is sufficient only if all such counterexample
queries are UNSAT. Every issued reason is a conjunction over original concepts,
not arbitrary hidden gate activations. No co-occurrence constraints enter the proof.

For joint calibration, the score is the maximum concept error probability across
coordinates. The finite-sample split-conformal quantile constructs a rectangular
set of plausible Boolean vectors. Supported reasons may use only literals fixed
throughout the nonempty set. Hence, if that set contains the true concept vector,
every issued premise is true. Under exchangeability and a procedure fixed before
calibration, `P(issued AND any false premise) <= alpha` marginally over calibration
and test draws. This is **not** error conditional on acceptance, per-image certainty,
task-label correctness, or a proof about which pixels the backbone uses. New image
perturbation regimes need an appropriate calibration design.

Search enumerates reasons by length with deterministic feature-order ties. It is
exponential in the number of candidate concepts in the worst case. The defaults
limit length to four, candidates to 5,000, and SAT conflicts to 100,000 per query.
`solver_budget`, `candidate_budget`, `empty_concept_set`, and
`no_reason_within_length` all abstain, and are saved separately. An incomplete search
does not prove that no explanation exists. When a reason is returned, all shorter
candidates within that policy have been checked. These are trusted-solver checks,
not independently replayable DRAT proof files.

## Outputs and experiments

Each run saves reader/head weights, wiring, an executable Boolean `circuit.json`,
train-only preprocessing, calibration scores, split IDs, configuration/package
versions, training history, metrics and per-image explanations. `infer.py` reloads
them and checks that the checkpoint and exported circuit agree. Ordinary and
supported reasons are compared on identical accepted images at equal lengths;
`matched_repairs` and `matched_spoils` separate this effect from abstention.

`--pyxai-check` constructs an exactly equivalent tree on the complete small input
cube and checks its sufficient reasons with the circuit solver. This retains PyXAI
as a cross-check; it does not claim PyXAI natively accepts arbitrary circuits.
The root `pilot.py` and original notebook remain available as the earlier two-gate
LTN/PyXAI baseline. DWN, LogicCBM, learned selection baselines and real-data region
perturbation experiments are not implemented here.

For a constant head, PyXAI 2.0.1 cannot enumerate reasons on a leaf-only tree.
The cross-check still verifies its predictions and checks the empty sufficient
reason with SAT; this separate count is explicitly labelled in the output.
PyXAI dependencies may print warnings about unrelated CLI flags or cleanup;
successful cross-check counts and process exit status determine completion.

For CVPR, first inspect oracle-concept accuracy, then the discrete head's quality,
then same-image explanation repairs. This package is infrastructure and a synthetic
smoke test; it does not establish improvement on CUB or a publication contribution.
