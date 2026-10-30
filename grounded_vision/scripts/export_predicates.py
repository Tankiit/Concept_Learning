"""
Script: export_predicates.py
============================
Performs grounded perception and exports structured FactDatabases.
Uses the fitted perception checkpoint (readers + prototype bank) and the
cached DINO features; supports ground-truth or predicted part regions.
"""

from __future__ import annotations
import os
import json
import argparse
import yaml
from tqdm import tqdm

from grounded_vision.pipeline import (
    active_vocabulary,
    build_backbone,
    build_dataset,
    load_perception,
    perceive_sample,
    FeatureProvider,
)
from grounded_vision.reasoning.features import RepresentationMode


def main():
    parser = argparse.ArgumentParser(description="Export grounded predicate databases.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--regions", type=str, default="gt", choices=["gt", "predicted"],
                        help="Part region source: ground-truth annotations or PartDetector predictions")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    attributes = active_vocabulary(cfg)
    dataset = build_dataset(cfg, args.split, args.limit or None)
    backbone = build_backbone(cfg)
    embed_dim = cfg["perception"]["embedding_dim"]
    part_detector, attr_reader, prototype_bank, meta = load_perception(cache_dir, embed_dim, attributes=attributes)
    if not meta["checkpoint_loaded"]:
        print("WARNING: exporting predicates with untrained readers; run fit_perception.py first.")
    provider = FeatureProvider(cache_dir, args.split, backbone=backbone)

    mode = RepresentationMode(cfg["experiment"].get("representation", "bound"))
    threshold = cfg["predicates"].get("probability_threshold", 0.5)

    out_file = os.path.join(cache_dir, f"{args.split}_predicates.jsonl")
    n = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)

    print(f"--> Exporting predicates for {n} samples from {args.split} split (regions={args.regions})...")
    n_missing = 0
    with open(out_file, "w") as f_out:
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
            record = {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "species_name": sample.species_name,
                "database": result.db.to_dict(),
            }
            f_out.write(json.dumps(record) + "\n")

    if n_missing:
        print(f"WARNING: {n_missing} samples were not in the feature cache and were skipped. "
              f"Re-run cache_features.py --split {args.split} with a larger --limit.")
    print(f"--> Successfully exported predicate databases to {out_file}")


if __name__ == "__main__":
    main()
