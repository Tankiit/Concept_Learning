"""
Script: cache_features.py
=========================
Extracts and caches frozen DINO backbone features for the CUB dataset.
"""

from __future__ import annotations
import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.backbone import FrozenDINOBackbone, extract_backbone_features


def main():
    parser = argparse.ArgumentParser(description="Cache frozen DINO features for CUB.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml", help="Path to YAML config")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=100, help="Max samples to cache for fast testing")
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

    for sp in splits:
        print(f"--> Caching features for split: {sp}")
        dataset = CUBDataset(
            data_dir=cfg["data"]["data_dir"],
            split=sp,
            image_size=cfg["data"]["image_size"],
            use_mock_if_missing=cfg["data"].get("use_mock_if_missing", True),
            num_mock_samples=args.limit,
        )

        n = min(len(dataset), args.limit)
        cls_tokens = []
        feature_maps = []
        labels = []
        sample_ids = []

        for i in tqdm(range(n), desc=f"Processing {sp}"):
            sample = dataset[i]
            img_tensor = sample.image_tensor.unsqueeze(0)
            feats = extract_backbone_features(backbone, img_tensor)

            cls_tokens.append(feats["cls_token"].cpu().numpy())
            # Downsample feature map if large or keep pooled
            feature_maps.append(feats["feature_map"].cpu().numpy())
            labels.append(sample.label)
            sample_ids.append(sample.sample_id)

        cls_arr = np.concatenate(cls_tokens, axis=0)
        fmap_arr = np.concatenate(feature_maps, axis=0)
        labels_arr = np.array(labels)
        ids_arr = np.array(sample_ids)

        out_path = os.path.join(cache_dir, f"{sp}_features.npz")
        np.savez_compressed(
            out_path,
            cls_tokens=cls_arr,
            feature_maps=fmap_arr,
            labels=labels_arr,
            sample_ids=ids_arr,
        )
        print(f"Saved {n} cached features to {out_path}")


if __name__ == "__main__":
    main()
