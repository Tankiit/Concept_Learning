"""
Script: train_head.py
=====================
Trains the certifiable decision tree head on compiled predicate features.
Uses the fitted perception checkpoint and cached DINO features; supports
ground-truth or predicted part regions.
"""

from __future__ import annotations
import os
import argparse
import pickle
import yaml
import numpy as np
from tqdm import tqdm

from grounded_vision.pipeline import (
    active_vocabulary,
    build_backbone,
    build_dataset,
    load_class_names,
    load_perception,
    perceive_sample,
    FeatureProvider,
)
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser(description="Train certifiable decision tree head.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Max training samples (0 = all)")
    parser.add_argument("--regions", type=str, default="gt", choices=["gt", "predicted"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    mode_str = cfg["experiment"].get("representation", "bound")
    mode = RepresentationMode(mode_str)
    attributes = active_vocabulary(cfg)
    compiler = FeatureCompiler(mode=mode, attributes=attributes)
    threshold = cfg["predicates"].get("probability_threshold", 0.5)

    print(f"--> Building training dataset for mode: {mode_str} (num_features: {compiler.num_features})")
    dataset = build_dataset(cfg, "train", args.limit or None)
    backbone = build_backbone(cfg)
    embed_dim = cfg["perception"]["embedding_dim"]
    part_detector, attr_reader, prototype_bank, pmeta = load_perception(cache_dir, embed_dim, attributes=attributes)
    if not pmeta["checkpoint_loaded"]:
        print("WARNING: training the head on predicates from untrained readers; run fit_perception.py first.")
    provider = FeatureProvider(cache_dir, "train", backbone=backbone)

    class_names = load_class_names(cfg["data"]["data_dir"]) or list(range(cfg["data"]["num_classes"]))

    dbs = []
    labels = []
    n = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)
    n_missing = 0

    print(f"--> Extracting grounded predicates for {n} training samples (regions={args.regions})...")
    for i in tqdm(range(n)):
        sample = dataset[i]
        if not provider.has_cache or sample.sample_id not in provider.index:
            n_missing += 1
            continue
        result = perceive_sample(
            sample, provider, part_detector, attr_reader, prototype_bank,
            region_mode=args.regions,
            probability_threshold=threshold,
            include_relations=(mode == RepresentationMode.RELATIONAL),
            attributes=attributes,
        )
        dbs.append(result.db)
        labels.append(sample.label)
    if n_missing:
        print(f"WARNING: {n_missing} training samples missing from feature cache were skipped.")
    print(f"--> Compiled {len(dbs)} grounded databases.")

    X_train = compiler.compile_batch(dbs, use_probabilities=False)
    y_train = np.array(labels)
    X_real = X_train
    y_real = y_train

    dcfg = cfg["reasoning"].get("distillation", {})
    if dcfg.get("enabled", False):
        seed = cfg["experiment"].get("seed", 42)
        teacher = RandomForestClassifier(
            n_estimators=dcfg.get("teacher_n_estimators", 100),
            max_depth=dcfg.get("teacher_max_depth", 12),
            min_samples_leaf=cfg["reasoning"].get("min_samples_leaf", 4),
            n_jobs=-1, random_state=seed,
        ).fit(X_train, y_train)
        rng = np.random.RandomState(seed)
        n_aug = dcfg.get("n_augment", 10)
        max_flips = dcfg.get("max_flips", 3)
        attr_cols = np.array([j for j, name in enumerate(compiler.feature_names) if not name.startswith("has_part(")])
        X_aug = np.repeat(X_real, n_aug, axis=0).copy()
        for i in range(len(X_aug)):
            k = rng.randint(1,max_flips+1)
            cols = rng.choice(attr_cols, k,replace = False)
            X_aug[i,cols] = 1 - X_aug[i,cols]
        y_aug = teacher.predict(X_aug)
        X_train = np.vstack([X_real, X_aug])
        y_train = np.concatenate([y_real,y_aug])

    tree_cfg = cfg["reasoning"]
    distill = dcfg.get("enabled",False)
    tree_head = CertifiableTreeHead(
        max_depth=dcfg.get("student_max_depth", 20) if distill else tree_cfg.get("max_depth", 10),
        min_samples_leaf=tree_cfg.get("min_samples_leaf", 3),
        criterion=tree_cfg.get("criterion", "gini"),
        feature_names=compiler.feature_names,
        class_names=class_names,
    )

    print("--> Fitting CertifiableTreeHead...")
    tree_head.fit(X_train, y_train)

    train_acc = float(np.mean(tree_head.predict(X_real) == y_real))
    print(f"--> Training Complete! Train Accuracy: {train_acc * 100:.2f}%")
    fidelity = None
    if distill:
        teacher_acc = float(np.mean(teacher.predict(X_real) == y_real))
        fidelity = float(np.mean(tree_head.predict(X_real) == teacher.predict(X_real)))
        print(f"--> Teacher train acc {teacher_acc * 100:.2f}% | "
            f"student fidelity to teacher {fidelity * 100:.2f}%")

    model_path = os.path.join(cache_dir, "tree_head.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "tree_head": tree_head,
                "compiler": compiler,
                "feature_names": compiler.feature_names,
                "class_names": class_names,
                "region_mode": args.regions,
                "distillation": dcfg if distill else None,
                "fidelity_train": fidelity,

            },
            f,
        )
    print(f"--> Model checkpoint saved to {model_path}")


if __name__ == "__main__":
    main()
