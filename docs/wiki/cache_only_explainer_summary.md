# Cache-only DINO explainer

**Date:** 2026-10-27  
**Scope:** 100 real CUB training images from `train_features.npz`.

- Classes represented: {np.int64(0): np.int64(30), np.int64(1): np.int64(30), np.int64(2): np.int64(30), np.int64(3): np.int64(10)}
- Training accuracy: 98.0%
- Mean certificate path length: 3.30 Boolean CLS-channel conditions
- Per-image certificates: `cache_only_explanations.jsonl`

## Interpretation

These are exact decision-tree path certificates for a cache-only DINO CLS-token baseline. They are **not grounded part/attribute explanations** and provide no spatial evidence. The Modal job is separately producing the intended predicate-grounded certificates.
