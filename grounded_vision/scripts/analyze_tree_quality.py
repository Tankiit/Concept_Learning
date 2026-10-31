"""
Script: analyze_tree_quality.py
===============================
Structural quality analysis of the SAVED tree checkpoints — no feature
caches or perception needed, everything comes from the fitted tree arrays.

Per tree: size/depth, training-weighted path length, leaf purity
distribution, impure-leaf count (leaves whose mixed classes make full-path
certificates vacuous), per-leaf sample masses (memorization indicator),
feature usage (counts + depth-weighted), unused concepts, class coverage.
Cross-tree: shared/unused concepts between the GT and predicted-region
CUB trees, and between the 28- and 312-concept heads.
"""

from __future__ import annotations
import json
import pickle
import numpy as np
from collections import Counter

TREES = {
    "cub28_gt": "/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_full/tree_head.pkl",
    "cub28_pred": "/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_full/tree_head_predicted_regions.pkl",
    "cub312_gt": "/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_312/tree_head.pkl",
    "celeba": "/Users/cril/tanmoy/research/data/grounded_vision_runs/celeba_concepts/tree_head.pkl",
}
OUT = "/Users/cril/tanmoy/research/data/grounded_vision_runs/tree_quality_report.json"


def analyze(name: str, path: str) -> dict:
    with open(path, "rb") as f:
        saved = pickle.load(f)
    head = saved["tree_head"]
    t = head.clf.tree_
    names = head.feature_names

    feature = t.feature
    left, right = t.children_left, t.children_right
    is_leaf = feature == -2
    n_nodes, n_leaves = t.node_count, int(is_leaf.sum())

    # node depths via BFS
    depth = np.zeros(n_nodes, dtype=int)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        if not is_leaf[node]:
            stack.append((left[node], d + 1))
            stack.append((right[node], d + 1))
    leaf_depths = depth[is_leaf]

    # training sample masses (n_node_samples is unweighted counts)
    samples = t.n_node_samples.astype(float)
    leaf_samples = samples[is_leaf]

    # leaf class distributions -> purity
    values = t.value[is_leaf, :, 0]                     # [n_leaves, n_classes]
    total = values.sum(axis=1, keepdims=True)
    purity = values.max(axis=1) / np.maximum(total, 1e-9)
    preds_leaf = values.argmax(axis=1)

    # feature usage: split counts and depth-weighted importance
    used = feature[~is_leaf]
    split_counts = Counter(used.tolist())
    depth_weighted = Counter()
    for node in range(n_nodes):
        if not is_leaf[node]:
            depth_weighted[int(feature[node])] += 1.0 / (depth[node] + 1)

    def fname(i):
        return names[i] if i < len(names) else f"feat_{i}"

    report = {
        "checkpoint": path,
        "n_nodes": int(n_nodes), "n_leaves": int(n_leaves),
        "max_depth": int(leaf_depths.max()),
        "mean_leaf_depth": float(leaf_depths.mean()),
        "train_weighted_mean_depth": float(
            (leaf_depths * leaf_samples).sum() / leaf_samples.sum()),
        "leaf_depth_p90": float(np.percentile(leaf_depths, 90)),
        "leaf_purity_mean": float(purity.mean()),
        "frac_leaves_pure_>=0.9": float((purity >= 0.9).mean()),
        "frac_leaves_pure_>=0.6": float((purity >= 0.6).mean()),
        "n_impure_leaves_<0.6": int((purity < 0.6).sum()),
        "median_leaf_samples": float(np.median(leaf_samples)),
        "p10_leaf_samples": float(np.percentile(leaf_samples, 10)),
        "n_leaves_min_samples": int((leaf_samples <= 2 * 1).sum()),
        "n_features_used": len(split_counts),
        "n_features_total": len(names),
        "top_features_by_splits": [
            {"feature": fname(i), "splits": c} for i, c in split_counts.most_common(8)],
        "top_features_depth_weighted": [
            {"feature": fname(i), "weight": round(w, 2)}
            for i, w in depth_weighted.most_common(8)],
        "n_classes_covered": int(len(set(preds_leaf.tolist()))),
    }
    # unused concepts (concept features only; skip has_part/relation features)
    unused = [n for i, n in enumerate(names)
              if i not in split_counts and (":" in n or n.startswith("exists"))]
    report["n_unused_concept_features"] = len(unused)
    report["unused_concepts_sample"] = unused[:8]
    return report


def main():
    reports = {}
    for name, path in TREES.items():
        try:
            reports[name] = analyze(name, path)
            r = reports[name]
            print(f"== {name}: {r['n_leaves']} leaves, depth {r['max_depth']}, "
                  f"purity {r['leaf_purity_mean']:.3f} (>=0.9: {r['frac_leaves_pure_>=0.9']*100:.0f}%), "
                  f"features {r['n_features_used']}/{r['n_features_total']}, "
                  f"median leaf n={r['median_leaf_samples']:.0f}")
        except Exception as e:
            reports[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"== {name}: ERROR {e}")

    # cross-tree structural comparison
    def used_set(key):
        r = reports.get(key, {})
        return {d["feature"] for d in r.get("top_features_by_splits", [])}

    if "cub28_gt" in reports and "cub28_pred" in reports and "error" not in reports["cub28_gt"]:
        with open(TREES["cub28_gt"], "rb") as f:
            gt_names = list(pickle.load(f)["tree_head"].feature_names)
        with open(TREES["cub28_pred"], "rb") as f:
            pr_used = {gt_names[i] for i in pickle.load(f)["tree_head"].clf.tree_.feature if i != -2}
        with open(TREES["cub28_gt"], "rb") as f:
            gt_used = {gt_names[i] for i in pickle.load(f)["tree_head"].clf.tree_.feature if i != -2}
        reports["cross_gt_vs_pred_regions"] = {
            "gt_used": len(gt_used), "pred_used": len(pr_used),
            "jaccard": len(gt_used & pr_used) / max(1, len(gt_used | pr_used)),
            "shared_top": sorted(gt_used & pr_used)[:10],
        }
        print(f"GT vs predicted-region trees: Jaccard of used concepts "
              f"{reports['cross_gt_vs_pred_regions']['jaccard']:.3f}")

    with open(OUT, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
