"""
Script: export_predicates.py
============================
Performs grounded perception and exports structured FactDatabases.
"""

from __future__ import annotations
import os
import json
import argparse
import yaml
from tqdm import tqdm

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.backbone import FrozenDINOBackbone, extract_backbone_features
from grounded_vision.perception.regions import extract_part_regions, pool_region_features
from grounded_vision.perception.readers import PartDetector, AttributeReader, PerceptionOutput
from grounded_vision.perception.prototypes import AttributePrototypeBank
from grounded_vision.predicates.grounding import ground_perception_output


def main():
    parser = argparse.ArgumentParser(description="Export grounded predicate databases.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    dataset = CUBDataset(
        data_dir=cfg["data"]["data_dir"],
        split=args.split,
        image_size=cfg["data"]["image_size"],
        use_mock_if_missing=True,
        num_mock_samples=args.limit,
    )

    backbone = FrozenDINOBackbone(
        model_name=cfg["perception"]["backbone"],
        embedding_dim=cfg["perception"]["embedding_dim"],
        patch_size=cfg["perception"]["patch_size"],
        image_size=cfg["data"]["image_size"],
    )

    embed_dim = cfg["perception"]["embedding_dim"]
    part_detector = PartDetector(embedding_dim=embed_dim)
    attr_reader = AttributeReader(embedding_dim=embed_dim)
    prototype_bank = AttributePrototypeBank(embedding_dim=embed_dim)

    out_file = os.path.join(cache_dir, f"{args.split}_predicates.jsonl")
    n = min(len(dataset), args.limit)

    print(f"--> Exporting predicates for {n} samples from {args.split} split...")
    with open(out_file, "w") as f_out:
        for i in tqdm(range(n)):
            sample = dataset[i]
            img_tensor = sample.image_tensor.unsqueeze(0)
            feats = extract_backbone_features(backbone, img_tensor)

            regions = extract_part_regions(sample.part_boxes, sample.part_visibilities)
            pooled_feats = pool_region_features(feats["feature_map"], regions)

            attr_probs, proto_matches = attr_reader.predict_all(
                pooled_feats, regions, prototype_bank
            )

            p_out = PerceptionOutput(
                sample_id=sample.sample_id,
                regions=regions,
                part_visibilities=sample.part_visibilities,
                attribute_probabilities=attr_probs,
                prototype_matches=proto_matches,
                feature_embeddings=pooled_feats,
            )

            db = ground_perception_output(
                p_out,
                probability_threshold=cfg["predicates"].get("probability_threshold", 0.5),
                include_relations=cfg["predicates"].get("include_relations", False),
                species_name=sample.species_name,
            )

            record = {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "species_name": sample.species_name,
                "database": db.to_dict(),
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"--> Successfully exported predicate databases to {out_file}")


if __name__ == "__main__":
    main()
