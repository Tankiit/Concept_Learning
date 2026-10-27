"""
Spatial Regions, Masks, and Anatomical Hierarchy
================================================
Defines spatial representations for bird parts, including bounding boxes,
binary/soft masks, spatial hierarchy (Whole Bird -> Macro Body Region -> Micro Part),
and RoI/attention pooling of frozen DINO patch features.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from ..data.cub import CUB_PARTS, CUB_PART_HIERARCHY, PART_TO_MACRO


@dataclass
class Region:
    """Represents a localized anatomical or spatial region in an image."""
    id: str
    part_name: str
    bbox: Tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) normalized [0.0, 1.0]
    confidence: float = 1.0
    parent_id: Optional[str] = None
    mask: Optional[np.ndarray] = None  # 2D boolean or float mask [H, W]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the (y_center, x_center) of the bounding box."""
        ymin, xmin, ymax, xmax = self.bbox
        return ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)

    @property
    def area(self) -> float:
        """Returns normalized bounding box area."""
        ymin, xmin, ymax, xmax = self.bbox
        return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)


class SpatialHierarchy:
    """
    Manages anatomical and spatial containment relationships:
    Whole Bird -> Macro Body Regions (Head, Torso, Appendages) -> Micro Anatomical Parts.
    """
    def __init__(self, regions: List[Region]):
        self.regions_by_id: Dict[str, Region] = {r.id: r for r in regions}
        self.regions_by_part: Dict[str, Region] = {r.part_name: r for r in regions}
        
    def get_part(self, part_name: str) -> Optional[Region]:
        return self.regions_by_part.get(part_name)

    def get_parent_macro(self, part_name: str) -> Optional[str]:
        return PART_TO_MACRO.get(part_name)

    def compute_iou(self, r1: Region, r2: Region) -> float:
        """Computes Intersection-over-Union between two normalized bounding boxes."""
        y1_min, x1_min, y1_max, x1_max = r1.bbox
        y2_min, x2_min, y2_max, x2_max = r2.bbox

        inter_ymin = max(y1_min, y2_min)
        inter_xmin = max(x1_min, x2_min)
        inter_ymax = min(y1_max, y2_max)
        inter_xmax = min(x1_max, x2_max)

        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_area = inter_h * inter_w

        union_area = r1.area + r2.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


def extract_part_regions(
    part_boxes: Dict[str, Tuple[float, float, float, float]],
    part_visibilities: Optional[Dict[str, float]] = None,
    min_confidence: float = 0.1,
) -> List[Region]:
    """
    Constructs a list of Region instances from part bounding boxes and visibilities.
    """
    regions: List[Region] = []
    for part_name, bbox in part_boxes.items():
        conf = 1.0
        if part_visibilities is not None:
            conf = float(part_visibilities.get(part_name, 1.0))
        if conf >= min_confidence:
            macro = PART_TO_MACRO.get(part_name, "body")
            regions.append(
                Region(
                    id=f"region_{part_name}",
                    part_name=part_name,
                    bbox=bbox,
                    confidence=conf,
                    parent_id=f"macro_{macro}",
                )
            )
    return regions


def pool_region_features(
    feature_map: torch.Tensor,
    regions: List[Region],
    pool_mode: str = "average",
) -> Dict[str, torch.Tensor]:
    """
    Pools backbone spatial feature map [1, D, H_g, W_g] over each region's bounding box.

    Args:
        feature_map: Tensor of shape [1, D, H_grid, W_grid]
        regions: List of Region instances with normalized coordinates [0.0, 1.0]
        pool_mode: "average" or "max"

    Returns:
        Dict mapping region.id -> pooled feature vector of shape [D]
    """
    assert feature_map.ndim == 4, f"Expected 4D feature map, got {feature_map.shape}"
    _, D, H_g, W_g = feature_map.shape
    device = feature_map.device

    pooled_features: Dict[str, torch.Tensor] = {}

    for r in regions:
        ymin, xmin, ymax, xmax = r.bbox
        # Map normalized coordinates [0, 1] to grid indices [0, H_g - 1]
        y_start = int(np.clip(np.floor(ymin * H_g), 0, H_g - 1))
        y_end = int(np.clip(np.ceil(ymax * H_g), y_start + 1, H_g))
        x_start = int(np.clip(np.floor(xmin * W_g), 0, W_g - 1))
        x_end = int(np.clip(np.ceil(xmax * W_g), x_start + 1, W_g))

        sub_map = feature_map[0, :, y_start:y_end, x_start:x_end]  # [D, H_sub, W_sub]
        if pool_mode == "average":
            vec = sub_map.mean(dim=(-2, -1))  # [D]
        elif pool_mode == "max":
            vec = sub_map.amax(dim=(-2, -1))  # [D]
        else:
            raise ValueError(f"Unknown pool mode: {pool_mode}")

        # L2-normalize region embedding
        vec = F.normalize(vec, p=2, dim=0)
        pooled_features[r.id] = vec

    return pooled_features
