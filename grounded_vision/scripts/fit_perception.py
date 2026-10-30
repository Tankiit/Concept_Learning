"""
Script: fit_perception.py
========================
Trains the part detector and attribute readers on cached DINO features,
then fits the attribute prototype bank from real positive exemplars.

Vectorized per-region training: all attribute heads sharing a grounding
region (part or whole-bird) are evaluated in one batched matrix multiply
per region, so the full CUB-312 vocabulary trains as fast as the
canonical 28 did with the old per-sample loop.

Supervision is fully real-CUB: part visibility and boxes from
parts/part_locs.txt, attribute labels from image_attribute_labels.txt
(aligned to the active vocabulary via data.attribute_set).
Requires {cache_dir}/train_features.npz (see cache_features.py).
"""

from __future__ import annotations
import os
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from grounded_vision.perception.regions import extract_part_regions, pool_region_features
from grounded_vision.perception.readers import PartDetector, AttributeReader
from grounded_vision.perception.prototypes import AttributePrototypeBank
from grounded_vision.pipeline import build_dataset, active_vocabulary
from grounded_vision.pipeline import FeatureProvider


def main():
    parser = argparse.ArgumentParser(description="Fit perception readers and prototypes.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--attr-weight", type=float, default=0.1,
                        help="Weight of the attribute loss in the joint objective")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    embed_dim = cfg["perception"]["embedding_dim"]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    print(f"--> Training perception heads on device: {device}")

    feats_path = os.path.join(cache_dir, "train_features.npz")
    if not os.path.exists(feats_path):
        raise FileNotFoundError(
            f"{feats_path} not found. Run: python scripts/cache_features.py --config {args.config} --split train"
        )
    d = np.load(feats_path)
    fmaps = d["feature_maps"]
    cls_tokens = torch.from_numpy(d["cls_tokens"].astype(np.float32)).to(device)
    cache_ids = [int(s) for s in d["sample_ids"]]
    n = len(cache_ids)

    attributes = active_vocabulary(cfg)
    dataset = build_dataset(cfg, "train", n)
    part_names = dataset.part_names
    id_to_idx = {}
    for i in range(len(dataset)):
        sid, _, _, _ = dataset.meta(i)
        id_to_idx[sid] = i
    n_attrs = len(attributes)
    print(f"--> Vocabulary: {n_attrs} bound attributes "
          f"({sum(1 for _, p, _, _ in attributes if p == 'bird')} whole-bird)")

    # ---- Assemble supervision tensors -------------------------------------
    vis_targets = np.zeros((n, len(part_names)), dtype=np.float32)
    box_targets = np.zeros((n, len(part_names), 4), dtype=np.float32)  # (ymin, xmin, h, w)
    box_mask = np.zeros((n, len(part_names)), dtype=np.float32)
    attr_labels = np.zeros((n, n_attrs), dtype=np.float32)

    # Per-region feature rows and attribute labels
    region_rows: dict = {}          # region -> list of (row, vec384)
    missing = 0
    for row, sid in enumerate(cache_ids):
        if sid not in id_to_idx:
            missing += 1
            continue
        _, boxes, vis, attrs = dataset.meta(id_to_idx[sid])
        attr_labels[row] = attrs
        fmap = torch.from_numpy(fmaps[row].astype(np.float32)).unsqueeze(0)
        regions = extract_part_regions(boxes, vis)
        pooled = pool_region_features(fmap, regions)
        by_part = {r.part_name: pooled[r.id].numpy() for r in regions if r.id in pooled}
        # whole-bird region: pooled over the entire patch map (bbox 0-1)
        bird_vec = F.normalize(fmap.mean(dim=(-2, -1)).squeeze(0), p=2, dim=0).numpy()
        by_part["bird"] = bird_vec
        for part, vec in by_part.items():
            region_rows.setdefault(part, []).append((row, vec))
        for pi, part in enumerate(part_names):
            vis_targets[row, pi] = float(vis.get(part, 0.0))
            if vis.get(part, 0.0) >= 0.5 and part in boxes:
                ymin, xmin, ymax, xmax = boxes[part]
                box_targets[row, pi] = (ymin, xmin, ymax - ymin, xmax - xmin)
                box_mask[row, pi] = 1.0
    if missing:
        print(f"WARNING: {missing} cached samples not found in dataset metadata; skipped supervision.")

    vis_t = torch.from_numpy(vis_targets).to(device)
    box_t = torch.from_numpy(box_targets).to(device)
    box_m = torch.from_numpy(box_mask).to(device)

    # Region design matrices + attribute columns
    region_tensors = {r: (torch.from_numpy(np.stack([v for _, v in rows])).to(device),
                          np.array([row for row, _ in rows]))
                      for r, rows in region_rows.items()}
    attrs_by_region: dict = {}
    for local_id, part, _, _ in attributes:
        attrs_by_region.setdefault(part, []).append(local_id)

    # ---- Models ------------------------------------------------------------
    part_detector = PartDetector(embedding_dim=embed_dim, part_names=part_names).to(device)
    attr_reader = AttributeReader(embedding_dim=embed_dim, attributes=attributes).to(device)
    optimizer = torch.optim.Adam(
        list(part_detector.parameters()) + list(attr_reader.parameters()), lr=args.lr
    )

    def region_weight_bias(region):
        ids = attrs_by_region[region]
        W = torch.cat([attr_reader.classifiers[f"attr_{a}"].weight for a in ids], dim=0)   # [k, D]
        b = torch.cat([attr_reader.classifiers[f"attr_{a}"].bias for a in ids], dim=0)     # [k]
        return W, b, ids

    # Precompute per-region labels and pos_weights (fixed across epochs)
    region_data = {}
    for region, (X, rows) in region_tensors.items():
        if region not in attrs_by_region:
            continue
        y = torch.from_numpy(attr_labels[rows][:, attrs_by_region[region]]).to(device)
        pos = y.sum(dim=0)
        neg = y.shape[0] - pos
        pos_weight = (neg / pos.clamp(min=1.0)).clamp(max=50.0)
        region_data[region] = (X, y, pos_weight)

    print(f"--> Training for {args.epochs} epochs on {n} cached samples, "
          f"{len(region_data)} regions...")
    for epoch in range(args.epochs):
        pres_logits = part_detector.presence_head(cls_tokens)
        loss_vis = F.binary_cross_entropy_with_logits(pres_logits, vis_t)

        raw_boxes = part_detector.bbox_head(cls_tokens).view(n, len(part_names), 4)
        diff = (raw_boxes - box_t).abs().mean(dim=-1)
        loss_box = (diff * box_m).sum() / box_m.sum().clamp(min=1.0)

        loss_attr_sum = torch.tensor(0.0, device=device)
        for region, (X, y, pos_weight) in region_data.items():
            W, b, ids = region_weight_bias(region)
            logits = X @ W.T + b                                    # [rows, k]
            loss_attr_sum = loss_attr_sum + F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight)

        loss = loss_vis + 0.5 * loss_box + args.attr_weight * loss_attr_sum / max(1, n_attrs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs} - loss {float(loss.item()):.4f} "
                  f"(vis {float(loss_vis.item()):.4f}, box {float(loss_box.item()):.4f}, "
                  f"attr {float(loss_attr_sum.item())/max(1,n_attrs):.4f})")

    # ---- Fit prototype bank from positive exemplars -------------------------
    k_protos = cfg["perception"].get("num_prototypes_per_attribute", 3)
    prototype_bank = AttributePrototypeBank(
        embedding_dim=embed_dim, num_prototypes_per_attr=k_protos, attributes=attributes
    )
    rng = np.random.RandomState(42)
    fitted_attrs = 0
    for local_id, part, attr_name, val in attributes:
        rows_list = region_rows.get(part)
        if not rows_list:
            continue
        positives = [vec for (row, vec) in rows_list if attr_labels[row, local_id] > 0.5]
        if not positives:
            continue
        idx = rng.choice(len(positives), size=min(k_protos, len(positives)), replace=False)
        exemplars = torch.from_numpy(np.stack([positives[j] for j in idx]))
        meta = [
            {"prototype_id": f"proto_{local_id}_{j}", "part": part, "attr": attr_name,
             "val": val, "exemplar_row": int(idx[j]), "index": j}
            for j in range(len(idx))
        ]
        prototype_bank.update_prototype_exemplars(local_id, exemplars, meta)
        fitted_attrs += 1
    print(f"--> Prototype bank fitted for {fitted_attrs}/{n_attrs} attributes.")

    # ---- Train metrics (per-region attribute accuracy) -----------------------
    part_detector.eval()
    attr_reader.eval()
    with torch.no_grad():
        vis_pred = (torch.sigmoid(part_detector.presence_head(cls_tokens)) > 0.5).float()
        vis_acc = float((vis_pred == vis_t).float().mean())
        per_region_acc = {}
        n_attr_correct, n_attr_total = 0.0, 0
        for region, (X, y, _) in region_data.items():
            W, b, ids = region_weight_bias(region)
            pred = (torch.sigmoid(X @ W.T + b) > 0.5).float()
            acc = float((pred == y).float().mean())
            per_region_acc[region] = acc
            n_attr_correct += acc * y.numel()
            n_attr_total += y.numel()

    metrics = {
        "n_train_samples": n,
        "n_attributes": n_attrs,
        "part_visibility_accuracy": vis_acc,
        "attribute_accuracy_train": n_attr_correct / max(1, n_attr_total),
        "attribute_accuracy_by_region": per_region_acc,
        "attributes_with_fitted_prototypes": fitted_attrs,
    }

    ckpt_path = os.path.join(cache_dir, "perception_checkpoint.pt")
    torch.save(
        {
            "part_detector": part_detector.state_dict(),
            "attr_reader": attr_reader.state_dict(),
            "prototype_bank": {int(a): p.cpu() for a, p in prototype_bank.prototypes.items()},
            "prototype_meta": {int(a): m for a, m in prototype_bank.metadata.items()},
            "metrics": metrics,
            "attribute_set": cfg["data"].get("attribute_set", "canonical"),
            "part_names": list(part_names),
            "n_attributes": n_attrs,
            "config_snapshot": {"epochs": args.epochs, "lr": args.lr, "k_prototypes": k_protos},
        },
        ckpt_path,
    )
    metrics_path = os.path.join(cache_dir, "perception_train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"--> Perception checkpoint saved to {ckpt_path}")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "attribute_accuracy_by_region"}, indent=2))


if __name__ == "__main__":
    main()
