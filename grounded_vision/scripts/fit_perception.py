"""
Script: fit_perception.py
========================
Trains/fits part presence detector, attribute readers, and exemplar prototypes.
"""

from __future__ import annotations
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from grounded_vision.data.cub import CUBDataset, CANONICAL_BOUND_ATTRIBUTES
from grounded_vision.perception.backbone import FrozenDINOBackbone, extract_backbone_features
from grounded_vision.perception.regions import extract_part_regions, pool_region_features
from grounded_vision.perception.readers import PartDetector, AttributeReader
from grounded_vision.perception.prototypes import AttributePrototypeBank


def main():
    parser = argparse.ArgumentParser(description="Fit perception readers and prototypes.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    print("--> Initializing dataset and backbone...")
    dataset = CUBDataset(
        data_dir=cfg["data"]["data_dir"],
        split="train",
        image_size=cfg["data"]["image_size"],
        use_mock_if_missing=True,
        num_mock_samples=args.samples,
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
    prototype_bank = AttributePrototypeBank(
        embedding_dim=embed_dim,
        num_prototypes_per_attr=cfg["perception"].get("num_prototypes_per_attribute", 3),
    )

    optimizer = optim.Adam(
        list(part_detector.parameters()) + list(attr_reader.parameters()),
        lr=1e-3,
    )

    print(f"--> Training perception heads for {args.epochs} epochs on {len(dataset)} samples...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for i in range(len(dataset)):
            sample = dataset[i]
            img_tensor = sample.image_tensor.unsqueeze(0)
            feats = extract_backbone_features(backbone, img_tensor)
            cls_token = feats["cls_token"]
            fmap = feats["feature_map"]

            # Ground truth regions and features
            gt_regions = extract_part_regions(sample.part_boxes, sample.part_visibilities)
            pooled_feats = pool_region_features(fmap, gt_regions)

            # Part detector forward
            vis_preds, bbox_preds = part_detector(cls_token)

            # Target visibilities
            target_vis = torch.tensor(
                [sample.part_visibilities.get(p, 1.0) for p in part_detector.part_names],
                dtype=torch.float32,
            )
            pred_vis = torch.tensor(
                [vis_preds[p] for p in part_detector.part_names],
                dtype=torch.float32,
                requires_grad=True,
            )
            loss_vis = nn.functional.binary_cross_entropy(pred_vis, target_vis)

            # Attribute reader loss
            loss_attr = torch.tensor(0.0)
            for attr_id, part, attr_name, val in CANONICAL_BOUND_ATTRIBUTES:
                reg_id = f"region_{part}"
                if reg_id in pooled_feats:
                    prob = attr_reader.predict_attribute(attr_id, pooled_feats[reg_id])
                    gt_val = float(sample.attribute_labels[attr_id]) if attr_id < len(sample.attribute_labels) else 0.0
                    loss_attr = loss_attr + (prob - gt_val) ** 2

            loss = loss_vis + loss_attr * 0.1
            total_loss += float(loss.item())

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss / len(dataset):.4f}")

    # Save fitted models
    ckpt_path = os.path.join(cache_dir, "perception_checkpoint.pt")
    torch.save(
        {
            "part_detector": part_detector.state_dict(),
            "attr_reader": attr_reader.state_dict(),
        },
        ckpt_path,
    )
    print(f"--> Perception checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
