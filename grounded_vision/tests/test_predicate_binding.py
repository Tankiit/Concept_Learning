"""
Unit Tests: Predicate Binding, Relations, and Provenance
========================================================
Verifies:
1. Entity creation and hierarchy
2. Provenance recording (bounding boxes, prototype matches)
3. Relational grounding (PartOf, Above, Below)
4. Domain theory mutual exclusion and prerequisite checks
"""

import pytest
import numpy as np
import torch

from grounded_vision.data.cub import CANONICAL_BOUND_ATTRIBUTES
from grounded_vision.perception.regions import Region, extract_part_regions
from grounded_vision.perception.readers import PerceptionOutput
from grounded_vision.perception.prototypes import PrototypeMatch
from grounded_vision.predicates.schema import (
    BirdEntity,
    PartEntity,
    Provenance,
    Fact,
    FactDatabase,
    PredicateType,
)
from grounded_vision.predicates.grounding import ground_perception_output
from grounded_vision.predicates.relations import compute_spatial_relation, extract_relations
from grounded_vision.reasoning.domain import DomainTheory, validate_fact_consistency


def test_entity_and_fact_creation():
    db = FactDatabase(sample_id=1)
    bird = BirdEntity(id="bird_0", species_name="001.Black_footed_Albatross")
    beak = PartEntity(id="part_beak_0", part_name="beak", parent_id="bird_0")
    db.add_entity(bird)
    db.add_entity(beak)

    prov = Provenance(
        region_id="region_beak",
        bbox=(0.1, 0.5, 0.3, 0.8),
        detector_confidence=0.95,
        reader_probability=0.88,
        prototype_id="proto_0_1",
        prototype_similarity=0.91,
    )
    fact = Fact(
        predicate="has_attribute",
        subject_id="part_beak_0",
        attribute="bill_shape",
        value="curved",
        confidence=0.88,
        truth_value=True,
        provenance=prov,
        predicate_type=PredicateType.HAS_ATTRIBUTE,
    )
    db.add_fact(fact)

    assert len(db.entities) == 2
    assert len(db.facts) == 1
    assert fact.key == "part_beak_0:bill_shape=curved"

    retrieved = db.get_fact("part_beak_0:bill_shape=curved")
    assert retrieved is not None
    assert retrieved.provenance is not None
    assert retrieved.provenance.prototype_id == "proto_0_1"
    assert retrieved.provenance.bbox == (0.1, 0.5, 0.3, 0.8)


def test_spatial_relation_computation():
    r_crown = Region(id="r_crown", part_name="crown", bbox=(0.05, 0.4, 0.2, 0.6))
    r_beak = Region(id="r_beak", part_name="beak", bbox=(0.25, 0.5, 0.35, 0.7))

    relations = compute_spatial_relation(r_crown, r_beak)
    rel_names = [r[0] for r in relations]

    assert "above" in rel_names
    assert "below" not in rel_names


def test_perception_grounding_pipeline():
    boxes = {
        "crown": (0.05, 0.4, 0.2, 0.6),
        "beak": (0.25, 0.5, 0.35, 0.7),
        "belly": (0.5, 0.4, 0.7, 0.6),
    }
    vis = {"crown": 1.0, "beak": 1.0, "belly": 1.0}
    regions = extract_part_regions(boxes, vis)

    attr_probs = {0: 0.85, 5: 0.90, 18: 0.20}
    proto_matches = {
        0: PrototypeMatch(
            prototype_id="proto_0_0",
            attribute_id=0,
            part_name="beak",
            attribute_name="bill_shape",
            value="curved",
            similarity=0.92,
        )
    }

    p_out = PerceptionOutput(
        sample_id=42,
        regions=regions,
        part_visibilities=vis,
        attribute_probabilities=attr_probs,
        prototype_matches=proto_matches,
        feature_embeddings={},
    )

    db = ground_perception_output(p_out, probability_threshold=0.5, include_relations=True)
    assert len(db.entities) >= 4  # bird + 3 parts
    active = db.get_active_facts()
    assert len(active) > 0

    # Check that curved bill fact exists and is True
    curved_facts = [f for f in active if f.attribute == "bill_shape" and f.value == "curved"]
    assert len(curved_facts) == 1
    assert curved_facts[0].provenance is not None
    assert curved_facts[0].provenance.prototype_id == "proto_0_0"


def test_domain_theory_mutual_exclusion():
    theory = DomainTheory()
    db = FactDatabase(sample_id=10)

    # Valid single color
    db.add_entity(PartEntity(id="part_beak_0", part_name="beak"))
    db.add_fact(
        Fact(
            predicate="has_attribute",
            subject_id="part_beak_0",
            attribute="bill_color",
            value="yellow",
            truth_value=True,
        )
    )
    violations = theory.check_mutual_exclusions(db)
    assert len(violations) == 0

    # Conflicting second color (both active)
    db.add_fact(
        Fact(
            predicate="has_attribute",
            subject_id="part_beak_0",
            attribute="bill_color",
            value="black",
            truth_value=True,
        )
    )
    violations = theory.check_mutual_exclusions(db)
    assert len(violations) == 1
    assert "Mutual exclusion violated" in violations[0]
