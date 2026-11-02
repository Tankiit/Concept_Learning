# Cache-only DINO explainer

**Date:** 2026-11-02  
**Scope:** 100 real CUB training images from `train_features.npz`.

- Classes represented: {0: 30, 1: 30, 2: 30, 3: 10}
- Training accuracy: 95.0%
- Mean certificate path length: 3.60 Boolean CLS-channel conditions
- Per-image certificates: `cache_only_explanations.jsonl`

## Interpretation

These are exact decision-tree path certificates for a cache-only DINO CLS-token baseline. They are **not grounded part/attribute explanations** and provide no spatial evidence. The Modal job is separately producing the intended predicate-grounded certificates.
