# Activation grounding and contrastive NLX

**Date:** 2026-10-27  
**Dataset:** 100 cached real CUB-200-2011 training images  
**Representation:** frozen DINO activation maps, 4×4 spatial pooling, 384 channels per tile

## What is certified

The activation tree predicts from pooled activation features. For each image, the explainer stores a minimal sufficient set of tree literals, and the independent tree verifier confirms that every compatible tree leaf has the same prediction.

| Metric | Result |
| --- | ---: |
| In-cache tree accuracy | 95.0% |
| Verified sufficient-reason rate | 100.0% |
| Mean sufficient-reason size | 3.8 conditions |

## Grounding protocol

Each literal is linked to a DINO channel, a 4×4 image tile, its 16×16 patch range, and image coordinates. A CUB anatomical part name is added **only** when a visible CUB part-location annotation lies inside that tile. Otherwise, the NLX retains the tile-only description.

This establishes spatial/part grounding. It does not yet establish that a channel represents a particular attribute such as *yellow belly*; that requires concept-mapping validation.

## Abductive explanation

For sample 2, the model predicts class 0 because the following conditions are sufficient:

> Activation channel 242 in tile row 4, column 3 is ≤ −0.314; and activation channel 68 in the annotated back, breast, left-wing, tail area is ≤ 8.764.

## Contrastive explanation

For the same sample, the nearest tree contrast is class 2:

> If activation channel 68 in tile row 2, column 2 were > 8.764 (observed: 7.151), the classifier could move toward class 2.

The contrast identifies a certified decision boundary condition, not an image-edit intervention. A later causal image intervention should test whether modifying the associated visual region changes this activation as predicted.

## Artifacts

- [`assets/activation_tree_grounded_examples.png`](assets/activation_tree_grounded_examples.png) — orange evidence tiles
- [`assets/activation_exemplar_examples.png`](assets/activation_exemplar_examples.png) — nearest exemplar prototype matches
- [`NLX_EXAMPLES.md`](NLX_EXAMPLES.md) — readable examples
- [`../../grounded_vision/results/activation_tree_grounded_metrics.json`](../../grounded_vision/results/activation_tree_grounded_metrics.json) — quantitative performance metrics
- [`../../grounded_vision/results/shap_vs_sufficient_results.json`](../../grounded_vision/results/shap_vs_sufficient_results.json) — TreeSHAP comparison data

## Next step

For each recurrent channel–tile condition, measure association with CUB attributes and parts on held-out images. Only mappings that pass a predefined reliability threshold should replace activation-channel wording in NLX.

## Part-only NLX policy

The `nlx_part_only` field says only that the verified evidence lies in annotated CUB parts, e.g. “The activation tree predicts class 0 from verified evidence in the back, breast, left-wing, and tail area.” If a tile contains no visible CUB part point, it says **“an unannotated image region.”** Do not guess a part or an attribute in that case.

## Counterfactual claims we can test

We can claim *model-level counterfactual sensitivity*: intervening on the exact pooled activation feature named by a contrastive condition crosses a tree decision boundary and changes its prediction. This is exact for the tree.

We cannot yet claim that editing an image part causes the predicted class to change. To make that claim, run region-level interventions: mask, blur, replace, or inpaint the grounded tile; re-extract DINO activations; and test whether the predicted activation change and class switch occur. Use matched control tiles and report intervention success, preservation of non-target evidence, and image-quality checks.

## How the tree is created

The cache contains frozen DINO feature maps of shape `[384, 16, 16]`. Each map is average-pooled into a 4×4 grid, yielding 6,144 activation features (`16 tiles × 384 channels`). A depth-6 CART decision tree (`min_samples_leaf=2`, Gini criterion, fixed seed 42) is fitted to the 100 CUB labels. PyXAI is available in the environment; the current starter adapter records that availability while the independent tree verifier certifies the sufficient and contrastive literals. A direct PyXAI solver integration remains the next adapter change.

## Attribution comparison: TreeSHAP

TreeSHAP was evaluated on the same activation decision tree and compared to the grounding-selected, verifier-sufficient reasons at matched explanation length. Mean overlap between TreeSHAP’s top-|reason| features and the selected sufficient reason was **88.6%**. TreeSHAP is an attribution baseline: it ranks features, but it does not certify sufficiency. The full per-image result is saved in `shap_vs_sufficient_results.json`.
