"""
Visual Evidence Grounding and Provenance Linking
================================================
Maps formal explanation conditions back to concrete visual evidence:
Image bounding boxes, localized part crops, and nearest prototype exemplars.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

from ..predicates.schema import FactDatabase, Fact
from .schema import ExplanationCondition, SufficientExplanation


@dataclass
class VisualEvidenceCard:
    """Concrete perceptual grounding for a single explanation condition."""
    condition_str: str
    feature_name: str
    part_name: Optional[str]
    bbox: Optional[Tuple[float, float, float, float]]  # (ymin, xmin, ymax, xmax)
    crop_image: Optional[Image.Image] = None
    confidence: float = 1.0
    prototype_id: Optional[str] = None
    prototype_similarity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_str": self.condition_str,
            "feature_name": self.feature_name,
            "part_name": self.part_name,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": float(self.confidence),
            "prototype_id": self.prototype_id,
            "prototype_similarity": float(self.prototype_similarity) if self.prototype_similarity else None,
        }


class EvidenceResolver:
    """
    Resolves explanation conditions back to FactDatabase facts and image crops.
    """
    def __init__(self, db: FactDatabase, raw_image: Optional[Image.Image] = None):
        self.db = db
        self.raw_image = raw_image

    def resolve_condition(self, condition: ExplanationCondition) -> VisualEvidenceCard:
        """Finds the underlying fact and spatial evidence for a condition."""
        f_name = condition.feature_name
        matched_fact: Optional[Fact] = None

        # Search in FactDatabase for matching fact
        for fact in self.db.facts.values():
            if ":" in f_name:
                # Bound attribute: e.g. "beak:bill_shape=curved"
                part_sub, attr_val = f_name.split(":", 1)
                if (
                    part_sub in fact.subject_id
                    and fact.attribute is not None
                    and f"{fact.attribute}={fact.value}" == attr_val
                ):
                    matched_fact = fact
                    break
            elif "has_part" in f_name:
                # Part presence: e.g. "has_part(beak)"
                p_target = f_name.replace("has_part(", "").replace(")", "").strip()
                if p_target in str(fact.value):
                    matched_fact = fact
                    break
            elif "=" in f_name:
                # Flat attribute: e.g. "bill_shape=curved"
                if fact.attribute is not None and f"{fact.attribute}={fact.value}" == f_name:
                    matched_fact = fact
                    break

        bbox: Optional[Tuple[float, float, float, float]] = None
        crop_img: Optional[Image.Image] = None
        proto_id: Optional[str] = None
        proto_sim: Optional[float] = None
        conf: float = condition.observed_value
        part_name: Optional[str] = None

        if matched_fact and matched_fact.provenance:
            prov = matched_fact.provenance
            bbox = prov.bbox
            proto_id = prov.prototype_id
            proto_sim = prov.prototype_similarity
            conf = prov.reader_probability
            part_name = matched_fact.subject_id.replace("part_", "").replace("_0", "")

            # Extract crop if image is provided
            if self.raw_image is not None and bbox is not None:
                W, H = self.raw_image.size
                ymin, xmin, ymax, xmax = bbox
                box_pixels = (
                    int(np.clip(xmin * W, 0, W - 1)),
                    int(np.clip(ymin * H, 0, H - 1)),
                    int(np.clip(xmax * W, 1, W)),
                    int(np.clip(ymax * H, 1, H)),
                )
                if box_pixels[2] > box_pixels[0] and box_pixels[3] > box_pixels[1]:
                    crop_img = self.raw_image.crop(box_pixels)

        return VisualEvidenceCard(
            condition_str=condition.condition_str,
            feature_name=condition.feature_name,
            part_name=part_name,
            bbox=bbox,
            crop_image=crop_img,
            confidence=conf,
            prototype_id=proto_id,
            prototype_similarity=proto_sim,
        )

    def resolve_sufficient_explanation(
        self, explanation: SufficientExplanation
    ) -> List[VisualEvidenceCard]:
        """Resolves visual evidence for all conditions in a sufficient explanation."""
        return [self.resolve_condition(c) for c in explanation.conditions]
