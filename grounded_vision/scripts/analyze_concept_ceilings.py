"""Accuracy anchors on the same concept vocabulary: where is the bottleneck?

Two heads on the SAME features the deployed tree sees:
1. multinomial logistic regression on PREDICTED concepts (train->test) —
   is a linear head on the same readers better than the interpretable tree?
2. decision tree (the deployed hyperparameters) on GT concept labels —
   the honest concept-vocabulary ceiling with a tree head.

Together with the deployed tree's accuracy these say whether the bottleneck
is perception (all three low) or the head (GT-tree >> predicted heads).
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from grounded_vision.pipeline import (
    active_vocabulary, build_backbone, build_dataset, load_perception,
    perceive_sample, FeatureProvider,
)
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode


def matrix(cfg, split, cache_dir, backbone, pd_, ar, pb, attributes, compiler, regions):
    provider = FeatureProvider(cache_dir, split, backbone=backbone)
    ds = build_dataset(cfg, split)
    X, y = [], []
    for i in range(len(ds.samples)):
        s = ds[i]
        res = perceive_sample(s, provider, pd_, ar, pb, region_mode=regions,
                              probability_threshold=0.5, include_relations=False,
                              attributes=attributes)
        X.append(compiler.compile_database(res.db, use_probabilities=False))
        y.append(s.label)
    return np.array(X), np.array(y)


def gt_matrix(cfg, split, attributes, compiler):
    ds = build_dataset(cfg, split)
    n = len(ds.samples)
    X = np.zeros((n, compiler.num_features), dtype=np.float32)
    y = np.zeros(n, dtype=int)
    name_to_col = {f"{p}:{a}={v}": j for j, (aid, p, a, v) in enumerate(attributes)}
    for i in range(n):
        sid, boxes, vis, labels = ds.meta(i)
        y[i] = ds.samples[i].label
        for j, (aid, p, a, v) in enumerate(attributes):
            X[i, name_to_col[f"{p}:{a}={v}"]] = labels[j]
        for col, name in enumerate(compiler.feature_names):
            if name.startswith("has_part("):
                part = name[9:-1]
                X[i, col] = 1 if vis.get(part, 0) > 0.5 else 0
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cub_bound_full_v2.yaml")
    ap.add_argument("--regions", default="gt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cache_dir = cfg["data"]["cache_dir"]
    attributes = active_vocabulary(cfg)
    compiler = FeatureCompiler(mode=RepresentationMode("bound"), attributes=attributes)
    backbone = build_backbone(cfg)
    pd_, ar, pb, _ = load_perception(cache_dir, cfg["perception"]["embedding_dim"],
                                     attributes=attributes)
    print("-- compiling predicted-concept matrices (train, test)...")
    Xtr, ytr = matrix(cfg, "train", cache_dir, backbone, pd_, ar, pb, attributes, compiler, args.regions)
    Xte, yte = matrix(cfg, "test", cache_dir, backbone, pd_, ar, pb, attributes, compiler, args.regions)

    print("-- logistic regression on predicted concepts...")
    logreg = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42).fit(Xtr, ytr)
    logreg_acc = float((logreg.predict(Xte) == yte).mean())

    print("-- GT-concept tree (deployed hyperparameters)...")
    Gtr, gtr_y = gt_matrix(cfg, "train", attributes, compiler)
    Gte, gte_y = gt_matrix(cfg, "test", attributes, compiler)
    tree = DecisionTreeClassifier(
        max_depth=cfg["reasoning"].get("max_depth", 12),
        min_samples_leaf=cfg["reasoning"].get("min_samples_leaf", 4),
        criterion=cfg["reasoning"].get("criterion", "gini"),
        random_state=42,
    ).fit(Gtr, gtr_y)
    gt_tree_acc = float((tree.predict(Gte) == gte_y).mean())
    gt_tree_train_acc = float((tree.predict(Gtr) == gtr_y).mean())

    print("-- logistic regression on GT concepts (label-noise control)...")
    gt_logreg = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42).fit(Gtr, gtr_y)
    gt_logreg_acc = float((gt_logreg.predict(Gte) == gte_y).mean())

    report = {
        "config": args.config,
        "regions": args.regions,
        "logreg_on_predicted_concepts": {"test_accuracy": logreg_acc},
        "logreg_on_gt_concepts": {"test_accuracy": gt_logreg_acc},
        "tree_on_gt_concepts": {
            "test_accuracy": gt_tree_acc,
            "train_accuracy": gt_tree_train_acc,
            "hyperparams": {"max_depth": tree.max_depth,
                            "min_samples_leaf": tree.min_samples_leaf},
        },
    }
    out = args.out or os.path.join(cache_dir, f"concept_ceilings_{args.regions}_regions.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
