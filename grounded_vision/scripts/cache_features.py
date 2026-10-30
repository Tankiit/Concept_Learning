"""
Script: cache_features.py
=========================
Extracts and caches frozen DINO backbone features for the CUB dataset.
Processes images in batches on the fastest available device (CUDA/MPS/CPU).
"""

from __future__ import annotations
import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

from grounded_vision.pipeline import build_dataset
from grounded_vision.perception.backbone import FrozenDINOBackbone, extract_backbone_features


def main():
    parser = argparse.ArgumentParser(description="Cache frozen DINO features for CUB.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml", help="Path to YAML config")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Max samples to cache (0 = entire split)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    splits = ["train", "test"] if args.split == "all" else [args.split]

    backbone = FrozenDINOBackbone(
        model_name=cfg["perception"]["backbone"],
        embedding_dim=cfg["perception"]["embedding_dim"],
        patch_size=cfg["perception"]["patch_size"],
        image_size=cfg["data"]["image_size"],
    )
    print(f"--> Backbone on device: {backbone.device} (mock={backbone.is_mock})")

    for sp in splits:
        print(f"--> Caching features for split: {sp}")
        dataset = build_dataset(cfg, sp, args.limit or None)

        n = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)
        cls_tokens = []
        feature_maps = []
        labels = []
        sample_ids = []

        for start in tqdm(range(0, n, args.batch_size), desc=f"Processing {sp}"):
            batch_idx = range(start, min(start + args.batch_size, n))
            images = torch.stack([dataset[i].image_tensor for i in batch_idx])
            feats = extract_backbone_features(backbone, images)

            cls_tokens.append(feats["cls_token"].detach().cpu().numpy())
            feature_maps.append(feats["feature_map"].detach().cpu().numpy())
            labels.extend(dataset[i].label for i in batch_idx)
            sample_ids.extend(dataset[i].sample_id for i in batch_idx)

        cls_arr = np.concatenate(cls_tokens, axis=0)
        fmap_arr = np.concatenate(feature_maps, axis=0)
        labels_arr = np.array(labels)
        ids_arr = np.array(sample_ids)

        out_path = os.path.join(cache_dir, f"{sp}_features.npz")
        np.savez(
            out_path,
            cls_tokens=cls_arr,
            feature_maps=fmap_arr,
            labels=labels_arr,
            sample_ids=ids_arr,
        )
        print(f"Saved {n} cached features to {out_path}")


if __name__ == "__main__":
    main()
