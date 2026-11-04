# Grounded NLX examples

> **Regenerated 2026-11-02** with the corrected tile pooling (the 2026-10-27 examples were based on row-strip "tiles"). Part names now cluster anatomically — head parts (beak, crown, forehead, throat) co-occur in one tile, back/wing parts in another — which is qualitative evidence that certified tiles are genuinely local. Held-out-split examples are in [`RESULTS.md`](RESULTS.md) §1.

The orange evidence tiles are shown in [activation_tree_grounded_examples.png](assets/activation_tree_grounded_examples.png). Each sentence below uses only a verified tree condition and either an exact tile or visible CUB part points inside that tile.

## Sample 2 — predicted class 0

> The activation tree predicts class 0 because activation channel 71 in an unannotated image region is > 2.148; and activation channel 39 in the beak, crown, forehead, nape, right eye, throat area is > 1.610; and activation channel 85 in an unannotated image region is <= 0.503.

## Sample 4 — predicted class 1

> The activation tree predicts class 1 because activation channel 71 in an unannotated image region is > 2.148; and activation channel 39 in the back, left wing, right wing area is > 1.610; and activation channel 85 in an unannotated image region is > 0.503; and activation channel 294 in an unannotated image region is <= -1.858.

## Sample 121 — predicted class 2

> The activation tree predicts class 2 because activation channel 71 in an unannotated image region is > 2.148; and activation channel 39 in the back area is > 1.610; and activation channel 85 in an unannotated image region is > 0.503; and activation channel 294 in the left wing area is > -1.858.

## Sample 181 — predicted class 3

> The activation tree predicts class 3 because activation channel 71 in an unannotated image region is <= 2.148; and activation channel 56 in the breast, nape, right wing, throat area is > -0.354; and activation channel 173 in an unannotated image region is > 0.667; and activation channel 271 in an unannotated image region is > 1.167; and activation channel 324 in the crown, forehead, right eye area is > 0.984.

## Scope

These are faithful template NLXs, not attribute-level descriptions. Part names are used only where CUB’s visible part-point annotation lies in the certified tile; tiles without annotations are stated as "an unannotated image region".

**What "faithful" means here:** faithful to the **model's computation** — every clause restates a certified condition over predicted concept values. If a concept reader misreads the image, the sentence is still faithful to the model and wrong about the bird. (The NL audit itself is lexical, not binding-based — roadmap Requirement 3.)

## The full concept→NLX chain, interactively

[`grounded_vision/notebooks/inspect_one_image.ipynb`](../../grounded_vision/notebooks/inspect_one_image.ipynb) walks one image through the complete chain **with stored outputs** (runs anywhere — mock backbone, no data download): perception (15 part regions, 28 bound-attribute readers) → grounded `FactDatabase` (74 facts with provenance) → demonstration tree → machine-checked sufficient reason (4 conditions, status VERIFIED) → evidence cards → the verbalized sentence → faithfulness audit (precision 1.00, recall 1.00, 0 unsupported claims). The rendered example:

> The model classifies this bird as 003.Sooty_Albatross because it observed the following certified sufficient conditions: (1) no dagger bill shape on the beak [confidence: 0.49], (2) no white breast color on the breast [confidence: 0.49], (3) no white crown color on the crown [confidence: 0.50], (4) the breast situated below the throat [confidence: 1.00]. These 4 conditions are mathematically sufficient under the deployed decision rules to guarantee this prediction.

Same caveat as the mock benchmark: mock backbone + untrained readers + a demonstration tree, so the *values* are illustrative of the machinery only (the notebook's intro states this). What it shows that the activation examples above cannot: the part→attribute→value→polarity clauses, the relational template ("the breast situated below the throat"), and the audit pass, all the way from pixels to sentence.
