"""
Shared Pipeline Utilities
=========================
Common plumbing for the pipeline scripts: device selection, perception
checkpoint loading (readers + prototype bank), cached-feature access,
region construction (ground-truth or predicted), and the single-image
perception-to-grounding routine shared by export/train/explain/evaluate.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch

from .data.cub import CUBDataset
from .data.attributes import active_vocabulary
from .perception.backbone import FrozenDINOBackbone, extract_backbone_features
from .perception.regions import Region, extract_part_regions, pool_region_features
from .perception.readers import PartDetector, AttributeReader, PerceptionOutput
from .perception.prototypes import AttributePrototypeBank
from .predicates.grounding import ground_perception_output

PERCEPTION_CHECKPOINT = "perception_checkpoint.pt"


def pick_device() -> str:
    """cuda > mps > cpu, matching Apple Silicon local runs."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_class_names(data_dir: str) -> Optional[List[str]]:
    """Reads the real 200 CUB class names from classes.txt, or None if absent."""
    classes_file = os.path.join(data_dir, "classes.txt")
    if not os.path.exists(classes_file):
        return None
    names: List[str] = []
    with open(classes_file, "r") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                name = parts[1].split(".", 1)[1] if parts[1][:3].isdigit() else parts[1]
                names.append(f"{int(parts[0]):03d}.{name}")
    return names or None


def build_backbone(cfg: dict, device: Optional[str] = None) -> FrozenDINOBackbone:
    return FrozenDINOBackbone(
        model_name=cfg["perception"]["backbone"],
        embedding_dim=cfg["perception"]["embedding_dim"],
        patch_size=cfg["perception"]["patch_size"],
        image_size=cfg["data"]["image_size"],
        device=device,
    )


def load_perception(
    cache_dir: str,
    embedding_dim: int = 384,
    device: str = "cpu",
    attributes=None,
) -> Tuple[PartDetector, AttributeReader, AttributePrototypeBank, Dict[str, Any]]:
    """
    Loads the fitted perception readers and prototype bank from
    {cache_dir}/perception_checkpoint.pt. Falls back to fresh (random)
    modules with an explicit warning when the checkpoint is missing, so
    smoke runs still work but silent-untrained reads are impossible.
    """
    ckpt_path = os.path.join(cache_dir, PERCEPTION_CHECKPOINT)
    meta: Dict[str, Any] = {"checkpoint_loaded": False, "path": ckpt_path}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) \
        if os.path.exists(ckpt_path) else None
    if ckpt is None:
        part_detector = PartDetector(embedding_dim=embedding_dim).to(device)
        attr_reader = AttributeReader(embedding_dim=embedding_dim, attributes=attributes).to(device)
        bank = AttributePrototypeBank(embedding_dim=embedding_dim, attributes=attributes)
        print(f"WARNING: no perception checkpoint at {ckpt_path}; using UNTRAINED readers.")
        return part_detector, attr_reader, bank, meta

    # Detector part ontology comes from the checkpoint (dataset-dependent)
    ckpt_parts = ckpt.get("part_names")
    if not ckpt_parts:
        n_ckpt = int(ckpt["part_detector"]["presence_head.2.weight"].shape[0])
        from .data.cub import CUB_PARTS as _PARTS
        ckpt_parts = _PARTS[:n_ckpt]
    part_detector = PartDetector(embedding_dim=embedding_dim, part_names=ckpt_parts).to(device)
    attr_reader = AttributeReader(embedding_dim=embedding_dim, attributes=attributes).to(device)
    bank = AttributePrototypeBank(embedding_dim=embedding_dim, attributes=attributes)
    part_detector.load_state_dict(ckpt["part_detector"])
    attr_reader.load_state_dict(ckpt["attr_reader"])
    if "prototype_bank" in ckpt:
        for attr_id, protos in ckpt["prototype_bank"].items():
            bank.prototypes[int(attr_id)] = protos.detach().float().cpu()
        if "prototype_meta" in ckpt and ckpt["prototype_meta"]:
            bank.metadata = {int(k): v for k, v in ckpt["prototype_meta"].items()}
        meta["prototypes_fitted"] = True
    part_detector.eval()
    attr_reader.eval()
    meta["checkpoint_loaded"] = True
    meta["train_metrics"] = ckpt.get("metrics", {})
    return part_detector, attr_reader, bank, meta


class FeatureProvider:
    """
    Serves per-sample backbone features from the {split}_features.npz cache,
    computing them with the live backbone only on cache misses.
    """

    def __init__(
        self,
        cache_dir: str,
        split: str,
        backbone: Optional[FrozenDINOBackbone] = None,
    ):
        self.cache_dir = cache_dir
        self.split = split
        self.backbone = backbone
        self.index: Dict[int, int] = {}
        path = os.path.join(cache_dir, f"{split}_features.npz")
        self.has_cache = os.path.exists(path)
        if self.has_cache:
            d = np.load(path)
            self.cls_tokens = d["cls_tokens"]
            self.feature_maps = d["feature_maps"]
            self.sample_ids = d["sample_ids"]
            self.index = {int(sid): i for i, sid in enumerate(self.sample_ids)}
            print(f"--> Feature cache loaded: {path} ({len(self.sample_ids)} samples)")
        else:
            print(f"--> No feature cache at {path}; will use live backbone for every sample.")

    def get(self, sample_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_token [1, D], feature_map [1, D, H, W])."""
        if self.has_cache and sample_id in self.index:
            i = self.index[sample_id]
            cls = torch.from_numpy(np.asarray(self.cls_tokens[i], dtype=np.float32)).unsqueeze(0)
            fmap = torch.from_numpy(np.asarray(self.feature_maps[i], dtype=np.float32)).unsqueeze(0)
            return cls, fmap
        if self.backbone is None:
            raise KeyError(
                f"sample {sample_id} not in {self.split} feature cache and no backbone provided"
            )
        raise KeyError(
            f"sample {sample_id} not in {self.split} feature cache; re-run cache_features.py "
            f"with a larger --limit or --split all"
        )


def regions_for_sample(
    sample,
    region_mode: str,
    part_detector: PartDetector,
    cls_token: torch.Tensor,
) -> List[Region]:
    """
    Builds the grounding regions for one sample.

    region_mode:
      - "gt":        ground-truth CUB part boxes (oracle upper bound)
      - "predicted": PartDetector visibility/box predictions from the CLS token
    """
    if region_mode == "gt":
        return extract_part_regions(sample.part_boxes, sample.part_visibilities)
    if region_mode == "predicted":
        vis_preds, box_preds = part_detector(cls_token)
        return extract_part_regions(box_preds, vis_preds)
    raise ValueError(f"Unknown region mode: {region_mode!r} (expected 'gt' or 'predicted')")


@dataclass
class PerceiveResult:
    db: Any
    output: PerceptionOutput
    vis_preds: Dict[str, float]
    box_preds: Dict[str, Tuple[float, float, float, float]]
    regions: List[Region]


def perceive_sample(
    sample,
    provider: FeatureProvider,
    part_detector: PartDetector,
    attr_reader: AttributeReader,
    prototype_bank: AttributePrototypeBank,
    region_mode: str = "gt",
    probability_threshold: float = 0.5,
    include_relations: bool = False,
    attributes=None,
) -> PerceiveResult:
    """Runs the shared perception -> predicate-grounding step for one sample."""
    from .data.cub import CANONICAL_BOUND_ATTRIBUTES
    if attributes is None:
        attributes = CANONICAL_BOUND_ATTRIBUTES
    cls_token, fmap = provider.get(sample.sample_id)
    regions = regions_for_sample(sample, region_mode, part_detector, cls_token)
    # Whole-bird vocabulary entries ground on a full-image region (subject bird_0)
    if any(part == "bird" for _, part, _, _ in attributes) and region_mode != "predicted":
        if not any(r.part_name == "bird" for r in regions):
            regions = regions + [Region(id="region_bird", part_name="bird",
                                        bbox=(0.0, 0.0, 1.0, 1.0), confidence=1.0)]
    pooled = pool_region_features(fmap, regions)
    attr_probs, proto_matches = attr_reader.predict_all(pooled, regions, prototype_bank)

    if region_mode == "predicted":
        vis_preds, box_preds = part_detector(cls_token)
        part_visibilities = vis_preds
    else:
        vis_preds, box_preds = part_detector(cls_token)
        part_visibilities = sample.part_visibilities
    part_visibilities = dict(part_visibilities)
    part_visibilities.setdefault("bird", 1.0)

    p_out = PerceptionOutput(
        sample_id=sample.sample_id,
        regions=regions,
        part_visibilities=part_visibilities,
        attribute_probabilities=attr_probs,
        prototype_matches=proto_matches,
        feature_embeddings=pooled,
    )
    db = ground_perception_output(
        p_out,
        probability_threshold=probability_threshold,
        include_relations=include_relations,
        species_name=sample.species_name,
        attributes=attributes,
    )
    return PerceiveResult(db=db, output=p_out, vis_preds=vis_preds, box_preds=box_preds, regions=regions)


def build_dataset(cfg: dict, split: str, limit: Optional[int] = None):
    """Constructs the configured dataset (CUB or CelebA) honoring mock policy
    and attribute set; returns a CUBSample-compatible Dataset."""
    data = cfg["data"]
    data_dir = data["data_dir"]
    if data.get("dataset") == "celeba":
        from .data.celeba import CelebADataset
        ms = data.get("max_samples", 0)
        if isinstance(ms, dict):
            ms = ms.get(split, 0)
        return CelebADataset(
            data_dir=data_dir,
            split={"train": "train", "test": "test"}.get(split, split),
            image_size=data["image_size"],
            target_attribute=data.get("target_attribute", "Male"),
            max_samples=ms or 0,
        )
    from .data.attributes import get_bound_attributes, vocabulary_cub_ids
    attribute_set = data.get("attribute_set", "canonical")
    has_attrs = os.path.exists(os.path.join(data_dir, "attributes.txt"))
    kwargs = {}
    if attribute_set != "canonical":
        kwargs = {
            "bound_attributes": get_bound_attributes(attribute_set, data_dir),
            "attribute_cub_ids": vocabulary_cub_ids(attribute_set, data_dir) if has_attrs else None,
        }
    from .data.cub import CUBDataset
    return CUBDataset(
        data_dir=data_dir,
        split=split,
        image_size=data["image_size"],
        use_mock_if_missing=data.get("use_mock_if_missing", True),
        num_mock_samples=limit if limit is not None else 120,
        **kwargs,
    )
