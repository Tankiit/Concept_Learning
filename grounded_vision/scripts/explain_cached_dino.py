"""Cache-only certified decision-tree explanations for a completed DINO cache."""
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

cache = Path("/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full/train_features.npz")
out = cache.parent / "cache_only_explanations.jsonl"
summary = cache.parent / "cache_only_explainer_summary.md"
d = np.load(cache)
x, y, ids = d["cls_tokens"], d["labels"], d["sample_ids"]

# Thresholding produces explicit Boolean features, which are suitable for
# path-level sufficient conditions in the fitted decision tree.
thresholds = np.median(x, axis=0)
xb = (x >= thresholds).astype(np.uint8)
tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, random_state=42)
tree.fit(xb, y)
pred = tree.predict(xb)
paths = tree.decision_path(xb)
features = tree.tree_.feature
threshold = tree.tree_.threshold

with out.open("w") as f:
    for row, (sample_id, target, prediction) in enumerate(zip(ids, y, pred)):
        nodes = paths.indices[paths.indptr[row]:paths.indptr[row + 1]]
        conditions = []
        for node in nodes:
            feat = int(features[node])
            if feat < 0:
                continue
            observed = int(xb[row, feat])
            direction = ">" if observed > threshold[node] else "<="
            conditions.append({"cls_channel": feat, "operator": direction,
                               "threshold": float(threshold[node]), "observed": observed})
        f.write(json.dumps({"sample_id": int(sample_id), "label": int(target),
                            "prediction": int(prediction), "conditions": conditions}) + "\n")

accuracy = float((pred == y).mean())
depths = [paths.indptr[i + 1] - paths.indptr[i] - 1 for i in range(len(y))]
summary.write_text(
    "# Cache-only DINO explainer\n\n"
    "**Date:** 2026-10-27  \n"
    "**Scope:** 100 real CUB training images from `train_features.npz`.\n\n"
    f"- Classes represented: {dict(zip(*np.unique(y, return_counts=True)))}\n"
    f"- Training accuracy: {accuracy:.1%}\n"
    f"- Mean certificate path length: {np.mean(depths):.2f} Boolean CLS-channel conditions\n"
    f"- Per-image certificates: `cache_only_explanations.jsonl`\n\n"
    "## Interpretation\n\n"
    "These are exact decision-tree path certificates for a cache-only DINO CLS-token baseline. "
    "They are **not grounded part/attribute explanations** and provide no spatial evidence. "
    "The Modal job is separately producing the intended predicate-grounded certificates.\n"
)
print(json.dumps({"images": len(y), "accuracy": accuracy, "mean_conditions": float(np.mean(depths)), "output": str(out)}))
