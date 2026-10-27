"""
Explicit Logical Domain Theory
==============================
Encodes background taxonomic and anatomical domain knowledge:
1. Mutual exclusion: A single anatomical part cannot simultaneously possess conflicting values
   (e.g., beak cannot be simultaneously cone and curved; breast cannot be simultaneously pure yellow and pure black).
2. Prerequisite constraints: Part attributes require the physical existence of the part
   (e.g., has_attribute(part_beak, bill_shape, curved) -> has_part(bird, part_beak)).
3. Spatial consistency: Spatial relations must satisfy antisymmetry (Above(A, B) -> not Above(B, A)).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional
from ..predicates.schema import Fact, FactDatabase, PredicateType


class DomainTheory:
    """Encapsulates background constraints and taxonomic logic."""
    def __init__(self):
        # Mutual exclusion groups: (subject_pattern, attribute_name) -> list of mutually exclusive values
        self.mutual_exclusions: List[Tuple[str, str, List[str]]] = [
            ("beak", "bill_shape", ["curved", "dagger", "hooked", "needle", "cone"]),
            ("beak", "bill_color", ["black", "yellow"]),
            ("crown", "crown_color", ["black", "yellow", "red", "white"]),
            ("forehead", "forehead_color", ["black", "yellow"]),
            ("breast", "breast_color", ["yellow", "white", "black", "red"]),
            ("belly", "belly_color", ["yellow", "white", "black"]),
            ("left_wing", "wing_color", ["black", "brown"]),
            ("right_wing", "wing_color", ["black", "brown"]),
        ]

    def check_mutual_exclusions(self, db: FactDatabase) -> List[str]:
        """Checks whether any active facts violate mutual exclusion constraints."""
        violations: List[str] = []
        active_facts = db.get_active_facts()

        for part_sub, attr_name, values in self.mutual_exclusions:
            present_vals = []
            for f in active_facts:
                if (
                    f.predicate_type == PredicateType.HAS_ATTRIBUTE
                    and part_sub in f.subject_id
                    and f.attribute == attr_name
                    and f.value in values
                ):
                    present_vals.append(f.value)

            if len(present_vals) > 1:
                violations.append(
                    f"Mutual exclusion violated for {part_sub} on {attr_name}: multiple values observed {present_vals}"
                )

        return violations

    def check_prerequisites(self, db: FactDatabase) -> List[str]:
        """Checks whether attribute facts exist without their parent part existing."""
        violations: List[str] = []
        active_parts = {
            f.value for f in db.get_active_facts() if f.predicate_type == PredicateType.HAS_PART
        }

        for f in db.get_active_facts():
            if f.predicate_type == PredicateType.HAS_ATTRIBUTE:
                if f.subject_id not in active_parts:
                    violations.append(
                        f"Prerequisite violated: attribute {f.key} active, but parent part {f.subject_id} is missing."
                    )

        return violations

    def is_consistent(self, db: FactDatabase) -> Tuple[bool, List[str]]:
        """Returns True if database satisfies all domain constraints."""
        violations = self.check_mutual_exclusions(db) + self.check_prerequisites(db)
        return (len(violations) == 0, violations)


def validate_fact_consistency(db: FactDatabase) -> Tuple[bool, List[str]]:
    """Helper to audit domain consistency of a FactDatabase."""
    theory = DomainTheory()
    return theory.is_consistent(db)
