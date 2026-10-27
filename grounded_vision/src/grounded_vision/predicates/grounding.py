"""
Perception Grounding Engine
===========================
Translates continuous neural reader outputs (part locations, visibilities,
attribute probabilities, and prototype similarities) into a structured FactDatabase
with explicit provenance.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

from ..data.cub import CANONICAL_BOUND_ATTRIBUTES
from ..perception.readers import PerceptionOutput
from .schema import (
    BirdEntity,
    PartEntity,
    Fact,
    FactDatabase,
    Provenance,
    PredicateType,
)
from .relations import extract_relations


def ground_perception_output(
    perception: PerceptionOutput,
    probability_threshold: float = 0.5,
    include_relations: bool = True,
    species_name: Optional[str] = None,
) -> FactDatabase:
    """
    Transforms perception outputs into an explicit, verifiable FactDatabase.

    Args:
        perception: PerceptionOutput containing regions, probabilities, and prototypes
        probability_threshold: Threshold for declaring binary truth value of attributes
        include_relations: Whether to compute and add anatomical and spatial relations
        species_name: Optional ground-truth or candidate species name

    Returns:
        Populated FactDatabase with entities, facts, and provenance
    """
    db = FactDatabase(sample_id=perception.sample_id)

    # 1. Instantiate root Bird entity
    bird = BirdEntity(id="bird_0", species_name=species_name)
    db.add_entity(bird)

    # 2. Instantiate Part entities and has_part facts
    part_entities: Dict[str, PartEntity] = {}
    region_by_part = {r.part_name: r for r in perception.regions}

    for r in perception.regions:
        part_name = r.part_name
        part_id = f"part_{part_name}_0"
        entity = PartEntity(id=part_id, part_name=part_name, parent_id="bird_0")
        db.add_entity(entity)
        part_entities[part_name] = entity

        # has_part fact
        part_vis = perception.part_visibilities.get(part_name, r.confidence)
        prov = Provenance(
            region_id=r.id,
            bbox=r.bbox,
            detector_confidence=r.confidence,
            reader_probability=part_vis,
        )
        has_part_fact = Fact(
            predicate="has_part",
            subject_id="bird_0",
            attribute=None,
            value=part_id,
            confidence=part_vis,
            truth_value=(part_vis >= probability_threshold),
            provenance=prov,
            predicate_type=PredicateType.HAS_PART,
        )
        db.add_fact(has_part_fact)

    # 3. Instantiate bound attribute facts
    for attr_id, part_name, attr_name, val in CANONICAL_BOUND_ATTRIBUTES:
        prob = perception.attribute_probabilities.get(attr_id, 0.0)
        truth = prob >= probability_threshold

        r = region_by_part.get(part_name)
        part_id = part_entities[part_name].id if part_name in part_entities else f"part_{part_name}_0"

        proto_match = perception.prototype_matches.get(attr_id)
        prov = None
        if r is not None:
            prov = Provenance(
                region_id=r.id,
                bbox=r.bbox,
                detector_confidence=r.confidence,
                reader_probability=prob,
                prototype_id=proto_match.prototype_id if proto_match else None,
                prototype_similarity=proto_match.similarity if proto_match else None,
            )

        fact = Fact(
            predicate="has_attribute",
            subject_id=part_id,
            attribute=attr_name,
            value=val,
            confidence=prob,
            truth_value=truth,
            provenance=prov,
            predicate_type=PredicateType.HAS_ATTRIBUTE,
        )
        db.add_fact(fact)

    # 4. Extract and add relational facts if enabled
    if include_relations:
        relational_facts = extract_relations(perception.regions, db)
        for rf in relational_facts:
            db.add_fact(rf)

    return db
