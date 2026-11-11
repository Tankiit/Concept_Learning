# CUB-200-2011

Not tracked.  Two inputs are needed:

1. **CUB root** (images + attribute annotations), official train/test
   split, nested checkout:
   `/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011`
   (must contain `images.txt`, `image_class_labels.txt`, attributes,
   `parts/`; restored `attributes.txt` provenance: fast.ai mirror, see
   the project wiki).

2. **DINOv3 feature cache** (flip-free eval transform, ViT-S/16):
   `/Users/tanmoy/research/data/grounded_vision_runs/cub_local_et`
   with `train_features.npz` / `test_features.npz`
   (keys: `cls_tokens`, `feature_maps` (N,384,16,16), `labels`,
   `sample_ids`).

Both also exist on the CRIL server; see `docs/wiki/` in the parent
repository for the Modal volume layout and the feature-extraction
recipe.
