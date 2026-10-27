# CUB bound pipeline — real-data run

**Date:** 2026-10-27  
**Status:** In progress — feature-cache stage completed; downstream stages have not yet produced completion artifacts.

## Purpose

Run the `grounded_vision` bound-representation starter pipeline on the local, official CUB-200-2011 data rather than its synthetic fallback.

## Data and configuration

| Item | Value |
| --- | --- |
| Dataset | CUB-200-2011 |
| Dataset root | `/Users/cril/tanmoy/research/data/CUB_200_2011/CUB_200_2011` |
| Real-data safeguard | `use_mock_if_missing: false` |
| Backbone | `dinov2_vits14` |
| Run config | `cub_bound_real_full.yaml` |
| Artifact root | `grounded_vision_runs/cub_bound_real_full` |

## Commands requested by the starter

1. `cache_features.py` — default: 100 training images
2. `fit_perception.py` — default: 60 images, 5 epochs
3. `export_predicates.py` — default: 50 test images
4. `train_head.py` — default: 100 training images
5. `explain.py --sample-idx 0`
6. `evaluate.py` — default: 40 test images

## Completed evidence

`train_features.npz` was created at `grounded_vision_runs/cub_bound_real_full/train_features.npz` from real CUB training images. Its expected DINO tensors are present: CLS tokens `[100, 384]` and patch feature maps `[100, 384, 16, 16]`.

## Pending artifacts

- `perception_checkpoint.pt`
- `test_predicates.jsonl`
- `tree_head.pkl`
- console explanation for test sample 0
- `results/benchmark_results.json`

## Note

The original repository configuration used a missing relative data path and allowed mock fallback. This run uses an isolated configuration with the explicit local CUB path and mock fallback disabled. Source files in the starter project were not changed.
