# Boolean predicates → deployed logic → PyXAI reasons → pixel regions

Start with `../boolean_cbm_walkthrough.ipynb`. This extension makes the CBM
interface explicit without replacing the existing LTN and difflogic experiments.
The demo uses synthetic coloured patches. It is not evidence about CUB accuracy.

## Run

Use the existing root `requirements.txt` (tested with PyXAI 2.0.1, PyTorch 2.8.0,
NumPy 2.3.5 and scikit-learn 1.8.0). From `ltn_pyxai_starter`:

```bash
python -m boolean_cbm.demo
python -m unittest boolean_cbm.test_boolean
```

## What each object means

| Object | Example | Meaning |
|---|---|---|
| Named predicate | `red_crown` | A supervised concept identity fixed before testing |
| Neural score | `q[0] = 0.83` | Reader output; not itself a Boolean fact or verified probability |
| Deployed bit | `c[0] = int(q[0] >= 0.5)` | Actual input to the hard head |
| Signed literal | `1` or `-1` | Predicted `red_crown` or predicted `NOT red_crown` |
| Sufficient reason | `(1, -3)` | These values force the head output for every other bit assignment |
| Region union | crops 1 and 3 | Both positive and negative literals contribute their entire support |

Unknown is not false. Missing annotations are excluded with `masked_concept_loss`.
The hard-head path requires a finite score for every predicate; NaNs are rejected.
This starter does not implement a three-valued logic head. A finite score on an
occluded part is still only a model prediction, not verified semantic evidence.

## Route A: start with a small exact Boolean rule

```python
from boolean_cbm import compile_boolean_function, threshold_scores

names = ['red_crown', 'long_beak', 'dark_wing']
def rule(c):
    return int(c[0] and (c[1] or not c[2]))

head = compile_boolean_function(rule, names)
bits = threshold_scores([.9, .8, .1])
reasons = head.reasons(bits, all_reasons=True)
for r in reasons:
    print(r, head.describe(r))
```

The two reasons are `(1, 2)` and `(1, -3)`. The formula is an illustrative
classifier, not an ornithological rule. The converter evaluates every Boolean
input and constructs an equivalent PyXAI tree by Shannon expansion. It checks
every exported prediction. Cost is exponential; the default limit is 10 concepts.
Never use full-cube export on a full CUB bottleneck.

## Route B: train a tree as the actual task head

```python
from boolean_cbm import fit_tree, threshold_scores
bits_train = threshold_scores(q_train)
head = fit_tree(bits_train, y_train, names, max_depth=6, min_samples_leaf=5)
head.save('head.json')
```

The tree is trained on **task labels and frozen predicted concepts**, and is the
deployed classifier. It is not a tree fitted to imitate a neural task head.
Its structural export is exact and scales with tree size, so it can use more
than 10 concepts. Depth and leaf-size controls trade accuracy against complexity;
tune them on a validation split and report test performance only after selection.
This implementation uses a fixed 0.5 predicate threshold. If thresholds are
tuned, do that before test access and persist them with the model.

Create an NPZ from your existing frozen CUB predicate scores:

```python
np.savez_compressed('cub_predicates.npz',
    q=probabilities, y=task_labels,
    split=np.asarray(split_names, dtype=str),
    ids=np.asarray(image_ids, dtype=str),
    groups=np.asarray(original_image_ids, dtype=str),
    concept_names=np.asarray(names, dtype=str))
```

`q` is `[N,M]`; `y`, `split`, `ids`, `groups` are `[N]`; names are `[M]`.
`split` permits train/tune/calibration/test. Only train fits the tree; only test
is reported. The reader must already be frozen without test leakage. Test rows
must have unique IDs; edits of the same source must share a group and split.

```bash
python -m boolean_cbm.train_tree --data cub_predicates.npz --out results/cub_tree
```

Outputs: portable `head.json`, `metrics.json`, and one checked explanation per
test image in `explanations.jsonl`. A constant head receives the valid empty
reason; accuracy reveals whether that trivial explanation is useful.

Inference requires no labels:

```python
from boolean_cbm import BooleanHead, threshold_scores
head, thresholds = BooleanHead.load('results/cub_tree/head.json')
c = threshold_scores(q_new, thresholds)
prediction = head.class_names[head.predict(c)]
reason = head.reasons(c)[0]
```

## Exact checks and PyXAI's role

The module constructs `Builder.DecisionTree` with
`force_features_equal_to_binaries=True`, then calls
`Explaining.initialize` and `sufficient_reason`. It verifies every returned
reason with a separate exact tree search:

1. The input satisfies every signed literal.
2. No compatible root-to-leaf path changes the prediction.
3. Removing any literal permits a different prediction.

The third property is subset minimality. It does not mean minimum length or
minimum pixel area. `all_reasons=True` can be exponential; use it only on small
examples. When all reasons are enumerated, you can compare their region-union
areas, but exact union area is not the sum of individual box areas.

In PyXAI 2.0.1, `is_sufficient_reason` uses sampled completions and is not the
exact verifier used here. `is_implicant` is an exact PyXAI tree query; our
independent tree traversal supplies an additional explicit check. Empty reasons
for constant heads are handled directly because PyXAI 2.0.1's leaf-only
enumeration has an edge-case failure. No timeout is interpreted as a proof.

## Real CUB execution (official split)

`../cub_crop_pipeline.py` runs all four steps on real CUB-200-2011 with the
repository's conventions: the 28 canonical bound attributes (one documented
substitution: `tail:tail_color=black` binds `has_upper_tail_color::black`),
template part boxes centred on CUB part points, the OFFICIAL train/test split,
readers and tree fit on train only (hyperparameters on a train-carved
validation sub-split), unknown-is-not-false masking (CUB certainty-1 guesses
and invisible-part crops are excluded from the reader loss), and one PyXAI +
independently verified sufficient reason per official-test image with each
literal judged against the CUB annotation.

```bash
python cub_crop_pipeline.py --classes 4 --out results/cub_k4_cropreader
python cub_crop_pipeline.py --classes 0 --out results/cub_full200_cropreader
```

Outputs: `head.json`, `reader.pt`, `metrics.json`, `per_attribute.json`,
`explanations.jsonl`, `provenance.json` (data SHA256, vocabulary, template
boxes, mask rule, deployment contract, spot-check that the vectorised scoring
path equals `CropPredicateReader.forward`). Readouts include task accuracy vs
the majority baseline, concept accuracy, reason length, per-literal and
all-literals-true rates, and part-visibility strata. These runs measure the
pipeline on real data; crop readers trained from scratch are the known
bottleneck and no grounding or causal claim is made.

## Crop-first reader and the pixel statement

`predicates.CropPredicateReader` takes `[N,3,H,W]` tensors and fixed integer
boxes `[N,M,4]` in `(y0,y1,x0,x1)` order (upper bounds exclusive). It slices
each original tensor before convolution, pooling or any other learned operation.
It returns logits. Train with `masked_concept_loss`, then deploy with
`reader.eval()` and `reader(...).sigmoid()` followed by fixed thresholds.

```python
reader = CropPredicateReader(n_predicates=len(names))
logits = reader(images, fixed_boxes)
loss = masked_concept_loss(logits, concept_labels, observed_mask)
loss.backward()
```

For this deterministic reader, with fixed boxes and weights, keeping all pixels
in a reason's region union fixed preserves those predicate values. Exact head
sufficiency then preserves the class label even if all other concept bits change.
This does not promise unchanged confidence/logits or correct semantic labels.
The guarantee begins at the supplied image tensor. A rerun part detector,
image-wide normalization, global attention before cropping, or batch-dependent
inference invalidates this simple dependency argument.

`region_union` is only geometry: calling it on an arbitrary full-image reader
does not certify that reader. Cached global DINO features pooled inside a box
support the concept-level route above, but do not meet the crop-first contract.
To study spatial certificates, recompute features from crops or prove the actual
encoder's dependency support. No CUB images, annotations, or reader weights are
bundled. Evaluate image-level concept correctness separately from class accuracy.

## Reuse the existing difflogic head

For <=10 concepts, compile its **hard** exported function exactly:

```python
from cvpr_logic.verify import predict
head = compile_boolean_function(
    lambda c: int(predict(circuit, np.asarray(c))[0]),
    circuit['concept_names'], [str(k) for k in range(circuit['classes'])])
```

For a larger circuit, use `cvpr_logic.verify.Explainer`: its gate CNF and class
comparison constraints reason about the deployed circuit without an exponential
tree expansion. PyXAI is then a small-instance cross-check, not a native arbitrary
neural-circuit interface. An LTN satisfaction objective, a hard logic circuit,
and a formal explanation are different components; no one implies the others.

## Primary documentation

- https://www.cril.univ-artois.fr/pyxai/documentation/learning/builder/decision_tree/
- https://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/sufficient_reason/
- https://www.cril.univ-artois.fr/pyxai/documentation/api/classes/explainerDT/
- https://github.com/crillab/pyxai
