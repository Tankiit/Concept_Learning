"""
Part and Attribute Readers
==========================
Perception heads that predict:
1. Part presence, localization, and visibility (PartDetector)
2. Fine-grained attributes conditioned on localized anatomical regions (AttributeReader)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.cub import CUB_PARTS, CANONICAL_BOUND_ATTRIBUTES
from .regions import Region, extract_part_regions, pool_region_features
from .prototypes import AttributePrototypeBank, PrototypeMatch


@dataclass
class PerceptionOutput:
    """Aggregated outputs from the perception subsystem for a single image."""
    sample_id: int
    regions: List[Region]
    part_visibilities: Dict[str, float]
    # attr_id -> predicted probability in [0.0, 1.0]
    attribute_probabilities: Dict[int, float]
    # attr_id -> PrototypeMatch
    prototype_matches: Dict[int, PrototypeMatch]
    feature_embeddings: Dict[str, torch.Tensor]  # region_id -> [D]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PartDetector(nn.Module):
    """
    Predicts anatomical part presence and bounding box offsets from CLS or global features.
    """
    def __init__(self, embedding_dim: int = 384, num_parts: int = 15,
                 part_names=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.part_names = list(part_names) if part_names is not None else CUB_PARTS[:num_parts]
        self.num_parts = len(self.part_names)

        # Multi-task head: part presence (logits) and normalized bbox (ymin, xmin, ymax, xmax)
        self.presence_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_parts),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_parts * 4),
            nn.Sigmoid(),
        )

    def forward(
        self, cls_token: torch.Tensor
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float, float, float]]]:
        """
        Forward pass for a single image embedding [1, D].

        Returns:
            visibilities: Dict mapping part_name -> presence probability in [0, 1]
            bboxes: Dict mapping part_name -> (ymin, xmin, ymax, xmax) in [0, 1]
        """
        assert cls_token.ndim == 2 and cls_token.shape[0] == 1
        pres_logits = self.presence_head(cls_token)[0]  # [num_parts]
        pres_probs = torch.sigmoid(pres_logits)

        raw_boxes = self.bbox_head(cls_token)[0].view(self.num_parts, 4)

        visibilities: Dict[str, float] = {}
        bboxes: Dict[str, Tuple[float, float, float, float]] = {}

        for i, part in enumerate(self.part_names):
            visibilities[part] = float(pres_probs[i].item())
            box = raw_boxes[i].detach().cpu().numpy()
            ymin, xmin = float(box[0]), float(box[1])
            ymax = float(np.clip(ymin + max(0.05, float(box[2])), ymin + 0.05, 1.0))
            xmax = float(np.clip(xmin + max(0.05, float(box[3])), xmin + 0.05, 1.0))
            bboxes[part] = (ymin, xmin, ymax, xmax)

        return visibilities, bboxes


class AttributeReader(nn.Module):
    """
    Predicts fine-grained attribute probabilities conditioned on localized part region embeddings.

    `attributes` is the active bound-attribute vocabulary (defaults to the
    canonical 28); entries may bind to the whole-bird region part "bird".
    """
    def __init__(self, embedding_dim: int = 384, attributes=None):
        super().__init__()
        from ..data.cub import CANONICAL_BOUND_ATTRIBUTES
        self.embedding_dim = embedding_dim
        self.attributes = list(attributes) if attributes is not None else list(CANONICAL_BOUND_ATTRIBUTES)
        # Dedicated linear probing head for each bound attribute
        self.classifiers = nn.ModuleDict()
        for attr_id, part, attr_name, val in self.attributes:
            key = f"attr_{attr_id}"
            self.classifiers[key] = nn.Linear(embedding_dim, 1)

    def predict_attribute(self, attr_id: int, region_feature: torch.Tensor) -> float:
        """
        Predicts probability of a single attribute given its corresponding region embedding.
        
        Args:
            attr_id: Canonical attribute index
            region_feature: Tensor of shape [D]
            
        Returns:
            Probability in [0.0, 1.0]
        """
        key = f"attr_{attr_id}"
        if key not in self.classifiers:
            return 0.5
        head = self.classifiers[key]
        logit = head(region_feature.unsqueeze(0)).squeeze()
        return float(torch.sigmoid(logit).item())

    def predict_all(
        self,
        region_features: Dict[str, torch.Tensor],
        regions: List[Region],
        prototype_bank: Optional[AttributePrototypeBank] = None,
    ) -> Tuple[Dict[int, float], Dict[int, PrototypeMatch]]:
        """
        Predicts all attributes across available regions and computes prototype matches.
        """
        attr_probs: Dict[int, float] = {}
        proto_matches: Dict[int, PrototypeMatch] = {}

        # Index region features by part_name ("bird" region included)
        feat_by_part: Dict[str, torch.Tensor] = {}
        for r in regions:
            if r.id in region_features:
                feat_by_part[r.part_name] = region_features[r.id]

        # One prototype-match per part (not per attribute): match_region already
        # returns every attribute bound to that part.
        matches_by_part: Dict[str, Dict[int, PrototypeMatch]] = {}

        for attr_id, part, attr_name, val in self.attributes:
            if part in feat_by_part:
                r_feat = feat_by_part[part]
                prob = self.predict_attribute(attr_id, r_feat)
                attr_probs[attr_id] = prob

                if prototype_bank is not None:
                    if part not in matches_by_part:
                        matches_by_part[part] = prototype_bank.match_region(r_feat, part_name=part)
                    match = matches_by_part[part].get(attr_id)
                    if match is not None:
                        proto_matches[attr_id] = match
            else:
                attr_probs[attr_id] = 0.0

        return attr_probs, proto_matches
