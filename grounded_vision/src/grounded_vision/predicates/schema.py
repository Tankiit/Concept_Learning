"""
Predicate and Entity Schema
===========================
Defines the relational ontology: Entities (Birds, Anatomical Parts),
Predicates (Part presence, bound attributes, relational connections),
Provenance tracking (spatial coordinates, confidence, prototype IDs),
and FactDatabases representing grounded scene states.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class PredicateType(str, Enum):
    HAS_PART = "has_part"
    HAS_ATTRIBUTE = "has_attribute"
    PART_OF = "part_of"
    SPATIAL = "spatial"


@dataclass
class Entity:
    """Base class for all grounded scene entities."""
    id: str
    entity_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BirdEntity(Entity):
    """The root bird individual."""
    species_name: Optional[str] = None

    def __init__(self, id: str = "bird_0", species_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(id=id, entity_type="bird", metadata=metadata or {})
        self.species_name = species_name


@dataclass
class PartEntity(Entity):
    """An anatomical part entity bound to an image region."""
    part_name: str = ""
    parent_id: str = "bird_0"

    def __init__(
        self,
        id: str,
        part_name: str,
        parent_id: str = "bird_0",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id=id, entity_type="part", metadata=metadata or {})
        self.part_name = part_name
        self.parent_id = parent_id


@dataclass
class Provenance:
    """Explicit grounding audit trail recording how a fact was perceived."""
    region_id: str
    bbox: Tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) normalized [0, 1]
    detector_confidence: float = 1.0
    reader_probability: float = 1.0
    prototype_id: Optional[str] = None
    prototype_similarity: Optional[float] = None
    patch_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """
    A grounded, verifiable first-order fact instance.
    Examples:
        has_part(bird_0, beak_0)
        has_attribute(beak_0, bill_shape, curved)
        above(crown_0, beak_0)
    """
    predicate: str
    subject_id: str
    attribute: Optional[str] = None
    value: Any = True
    confidence: float = 1.0
    truth_value: bool = True
    provenance: Optional[Provenance] = None
    predicate_type: PredicateType = PredicateType.HAS_ATTRIBUTE

    @property
    def key(self) -> str:
        """Returns a canonical string representation for indexing and feature compilation."""
        if self.predicate_type == PredicateType.HAS_ATTRIBUTE:
            return f"{self.subject_id}:{self.attribute}={self.value}"
        elif self.predicate_type == PredicateType.HAS_PART:
            return f"has_part({self.subject_id}, {self.value})"
        elif self.predicate_type in (PredicateType.PART_OF, PredicateType.SPATIAL):
            return f"{self.predicate}({self.subject_id}, {self.value})"
        return f"{self.predicate}({self.subject_id}, {self.attribute}={self.value})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "predicate": self.predicate,
            "subject_id": self.subject_id,
            "attribute": self.attribute,
            "value": str(self.value),
            "confidence": float(self.confidence),
            "truth_value": bool(self.truth_value),
            "provenance": {
                "region_id": self.provenance.region_id,
                "bbox": list(self.provenance.bbox),
                "confidence": self.provenance.detector_confidence,
                "prototype_id": self.provenance.prototype_id,
                "similarity": self.provenance.prototype_similarity,
            }
            if self.provenance
            else None,
        }


class FactDatabase:
    """
    In-memory knowledge base containing all grounded entities and facts for an image.
    Provides fast lookup by subject, attribute, part, and truth status.
    """
    def __init__(self, sample_id: int):
        self.sample_id = sample_id
        self.entities: Dict[str, Entity] = {}
        self.facts: Dict[str, Fact] = {}

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_fact(self, fact: Fact) -> None:
        self.facts[fact.key] = fact

    def get_fact(self, key: str) -> Optional[Fact]:
        return self.facts.get(key)

    def query_facts(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        attribute: Optional[str] = None,
        truth_value: Optional[bool] = None,
    ) -> List[Fact]:
        """Queries facts matching given filter criteria."""
        results = []
        for f in self.facts.values():
            if subject_id is not None and f.subject_id != subject_id:
                continue
            if predicate is not None and f.predicate != predicate:
                continue
            if attribute is not None and f.attribute != attribute:
                continue
            if truth_value is not None and f.truth_value != truth_value:
                continue
            results.append(f)
        return results

    def get_active_facts(self) -> List[Fact]:
        """Returns all facts where truth_value is True."""
        return [f for f in self.facts.values() if f.truth_value]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "entities": {eid: e.entity_type for eid, e in self.entities.items()},
            "facts": [f.to_dict() for f in self.facts.values()],
        }
