"""
PartOf and Spatial Relations
============================
Extracts relational predicates:
1. Anatomical hierarchy: PartOf(micro_part, macro_part), PartOf(macro_part, bird)
2. Directional spatial relations: Above, Below, LeftOf, RightOf, AdjacentTo
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from ..data.cub import PART_TO_MACRO, CUB_PART_HIERARCHY
from ..perception.regions import Region
from .schema import Fact, FactDatabase, Provenance, PredicateType


def compute_spatial_relation(
    r1: Region, r2: Region, delta_coord: float = 0.05
) -> List[Tuple[str, float]]:
    """
    Computes spatial relations between two regions r1 and r2.
    Returns list of (relation_name, confidence).
    """
    relations: List[Tuple[str, float]] = []
    y1, x1 = r1.center
    y2, x2 = r2.center

    # Vertical relations (smaller y is higher up in standard image coords)
    if y1 < y2 - delta_coord:
        confidence = float(np.clip((y2 - y1) / 0.5, 0.5, 1.0))
        relations.append(("above", confidence))
    elif y1 > y2 + delta_coord:
        confidence = float(np.clip((y1 - y2) / 0.5, 0.5, 1.0))
        relations.append(("below", confidence))

    # Horizontal relations
    if x1 < x2 - delta_coord:
        confidence = float(np.clip((x2 - x1) / 0.5, 0.5, 1.0))
        relations.append(("left_of", confidence))
    elif x1 > x2 + delta_coord:
        confidence = float(np.clip((x1 - x2) / 0.5, 0.5, 1.0))
        relations.append(("right_of", confidence))

    # Adjacency / Proximity
    center_dist = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
    if center_dist < 0.35:
        confidence = float(np.clip(1.0 - (center_dist / 0.35), 0.5, 1.0))
        relations.append(("adjacent_to", confidence))

    return relations


def extract_relations(regions: List[Region], db: FactDatabase) -> List[Fact]:
    """
    Extracts all anatomical PartOf and spatial relations across active regions.
    """
    relational_facts: List[Fact] = []
    region_by_part = {r.part_name: r for r in regions}

    # 1. Anatomical PartOf relations
    for part_name, r in region_by_part.items():
        macro = PART_TO_MACRO.get(part_name, "torso")
        part_id = f"part_{part_name}_0"
        macro_id = f"macro_{macro}_0"

        prov = Provenance(
            region_id=r.id,
            bbox=r.bbox,
            detector_confidence=r.confidence,
        )
        fact_micro = Fact(
            predicate="part_of",
            subject_id=part_id,
            attribute=None,
            value=macro_id,
            confidence=1.0,
            truth_value=True,
            provenance=prov,
            predicate_type=PredicateType.PART_OF,
        )
        relational_facts.append(fact_micro)

    # 2. Key spatial relations between anatomical neighbors
    # (e.g. crown above beak, throat below beak, belly below breast)
    salient_pairs = [
        ("crown", "beak"),
        ("crown", "left_eye"),
        ("crown", "right_eye"),
        ("throat", "beak"),
        ("breast", "throat"),
        ("belly", "breast"),
        ("back", "breast"),
        ("tail", "back"),
    ]

    for p1, p2 in salient_pairs:
        if p1 in region_by_part and p2 in region_by_part:
            r1 = region_by_part[p1]
            r2 = region_by_part[p2]
            computed = compute_spatial_relation(r1, r2)
            for rel_name, conf in computed:
                fact = Fact(
                    predicate=rel_name,
                    subject_id=f"part_{p1}_0",
                    attribute=None,
                    value=f"part_{p2}_0",
                    confidence=conf,
                    truth_value=(conf >= 0.5),
                    provenance=Provenance(
                        region_id=f"{r1.id}_{r2.id}",
                        bbox=r1.bbox,
                        detector_confidence=min(r1.confidence, r2.confidence),
                    ),
                    predicate_type=PredicateType.SPATIAL,
                )
                relational_facts.append(fact)

    return relational_facts
