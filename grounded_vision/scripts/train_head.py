"""
Script: train_head.py
=====================
Trains the certifiable decision tree head on compiled predicate features.
"""

from __future__ import annotations
import os
import argparse
import pickle
import yaml
import numpy as np

from grounded_vision.data.cub import CUBDataset, DEFAULT_SPECIES
from grounded_vision.perception.backbone import FrozenDINOBackbone, extract_backbone_features
from grounded_vision.perception.regions import extract_part_regions, pool_region_features
from grounded_vision.perception.readers import PartDetector, AttributeReader, PerceptionOutput
from grounded_vision.perception.prototypes import AttributePrototypeBank
from grounded_vision.predicates.grounding import ground_perception_output
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead


def main():
    parser = argparse.ArgumentParser(description="Train certifiable decision tree head.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    mode_str = cfg["experiment"].get("representation", "bound")
    mode = RepresentationMode(mode_str)
    compiler = FeatureCompiler(mode=mode)

    print(f"--> Building training dataset for mode: {mode_str} (num_features: {compiler.num_features})")
    dataset = CUBDataset(
        data_dir=cfg["data"]["data_dir"],
        split="train",
        image_size=cfg["data"]["image_size"],
        use_mock_if_missing=True,
        num_mock_samples=args.num_samples,
    )

    backbone = FrozenDINOBackbone(
        model_name=cfg["perception"]["backbone"],
        embedding_dim=cfg["perception"]["embedding_dim"],
        patch_size=cfg["perception"]["patch_size"],
        image_size=cfg["data"]["image_size"],
    )

    embed_dim = cfg["perception"]["embedding_dim"]
    attr_reader = AttributeReader(embedding_dim=embed_dim)
    prototype_bank = AttributePrototypeBank(embedding_dim=embed_dim)

    dbs = []
    labels = []

    print(f"--> Extracting grounded predicates for {len(dataset)} training samples...")
    for i in range(len(dataset)):
        sample = dataset[i]
        feats = extract_backbone_features(backbone, sample.image_tensor.unsqueeze(0))
        regions = extract_part_regions(sample.part_boxes, sample.part_visibilities)
        pooled = pool_region_features(feats["feature_map"], regions)
        attr_probs, proto_matches = attr_reader.predict_all(pooled, regions, prototype_bank)

        p_out = PerceptionOutput(
            sample_id=sample.sample_id,
            regions=regions,
            part_visibilities=sample.part_visibilities,
            attribute_probabilities=attr_probs,
            prototype_matches=proto_matches,
            feature_embeddings=pooled,
        )

        db = ground_perception_output(
            p_out,
            probability_threshold=cfg["predicates"].get("probability_threshold", 0.5),
            include_relations=(mode == RepresentationMode.RELATIONAL),
            species_name=sample.species_name,
        )
        dbs.append(db)
        labels.append(sample.label)

    X_train = compiler.compile_batch(dbs, use_probabilities=False)
    y_train = np.array(labels)

    tree_cfg = cfg["reasoning"]
    tree_head = CertifiableTreeHead(
        max_depth=tree_cfg.get("max_depth", 10),
        min_samples_leaf=tree_cfg.get("min_samples_leaf", 3),
        criterion=tree_cfg.get("criterion", "gini"),
        feature_names=compiler.feature_names,
        class_names=DEFAULT_SPECIES,
    )

    print("--> Fitting CertifiableTreeHead...")
    tree_head.fit(X_train, y_train)

    train_acc = float(np.mean(tree_head.predict(X_train) == y_train))
    print(f"--> Training Complete! Train Accuracy: {train_acc * 100:.2f}%")

    model_path = os.path.join(cache_dir, "tree_head.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "tree_head": tree_head,
                "compiler": compiler,
                "feature_names": compiler.feature_names,
                "class_names": DEFAULT_SPECIES,
            },
            f,
        )
    print(f"--> Model checkpoint saved to {model_path}")


if __name__ == "__main__":
    main()
