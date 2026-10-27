"""
Feature Compiler: Facts to Classifier Inputs
============================================
Compiles grounded FactDatabases into structured numerical feature vectors
supporting three distinct abstraction levels:
1. Flat: Global attribute presence (traditional CBM baseline)
2. Bound: Entity-bound part attributes (Part-Attribute-Value)
3. Relational: Bound attributes + Anatomical PartOf + Spatial Relations
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np

from ..data.cub import CANONICAL_BOUND_ATTRIBUTES
from ..predicates.schema import FactDatabase, Fact, PredicateType


class RepresentationMode(str, Enum):
    FLAT = "flat"
    BOUND = "bound"
    RELATIONAL = "relational"


class FeatureCompiler:
    """
    Compiles grounded facts into numerical arrays and maintains a 1-to-1
    invertible feature dictionary for interpretable tree reasoning.
    """
    def __init__(self, mode: RepresentationMode = RepresentationMode.BOUND):
        self.mode = mode
        self.feature_names: List[str] = []
        self.feature_to_idx: Dict[str, int] = {}
        self.idx_to_feature: Dict[int, str] = {}
        self._build_feature_vocabulary()

    def _build_feature_vocabulary(self) -> None:
        """Constructs canonical feature order depending on representation mode."""
        names: List[str] = []

        if self.mode == RepresentationMode.FLAT:
            # Global attributes: e.g. "bill_shape=curved", "breast_color=yellow"
            for _, _, attr, val in CANONICAL_BOUND_ATTRIBUTES:
                feat = f"{attr}={val}"
                if feat not in names:
                    names.append(feat)

        elif self.mode == RepresentationMode.BOUND:
            # Bound attributes: e.g. "beak:bill_shape=curved"
            for _, part, attr, val in CANONICAL_BOUND_ATTRIBUTES:
                feat = f"{part}:{attr}={val}"
                if feat not in names:
                    names.append(feat)
            # Add part presence features: e.g. "has_part(beak)"
            canonical_parts = sorted(list(set(part for _, part, _, _ in CANONICAL_BOUND_ATTRIBUTES)))
            for p in canonical_parts:
                feat = f"has_part({p})"
                if feat not in names:
                    names.append(feat)

        elif self.mode == RepresentationMode.RELATIONAL:
            # Bound attributes + part presence
            for _, part, attr, val in CANONICAL_BOUND_ATTRIBUTES:
                feat = f"{part}:{attr}={val}"
                if feat not in names:
                    names.append(feat)
            canonical_parts = sorted(list(set(part for _, part, _, _ in CANONICAL_BOUND_ATTRIBUTES)))
            for p in canonical_parts:
                feat = f"has_part({p})"
                if feat not in names:
                    names.append(feat)

            # Spatial & anatomical relations
            salient_relations = [
                "above(crown, beak)",
                "above(crown, left_eye)",
                "above(crown, right_eye)",
                "below(throat, beak)",
                "below(breast, throat)",
                "below(belly, breast)",
                "below(tail, back)",
                "adjacent_to(beak, forehead)",
                "adjacent_to(breast, belly)",
            ]
            for rel in salient_relations:
                if rel not in names:
                    names.append(rel)

        self.feature_names = sorted(names)
        self.feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.idx_to_feature = {i: name for i, name in enumerate(self.feature_names)}

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def compile_database(
        self, db: FactDatabase, use_probabilities: bool = False
    ) -> np.ndarray:
        """
        Compiles a single FactDatabase into a 1D vector [num_features].

        Args:
            db: Grounded FactDatabase
            use_probabilities: If True, uses continuous confidence in [0, 1];
                              otherwise uses binary truth value in {0.0, 1.0}.
        """
        vec = np.zeros(self.num_features, dtype=np.float32)

        for fact in db.facts.values():
            val = float(fact.confidence) if use_probabilities else (1.0 if fact.truth_value else 0.0)

            if self.mode == RepresentationMode.FLAT:
                if fact.predicate_type == PredicateType.HAS_ATTRIBUTE:
                    feat_key = f"{fact.attribute}={fact.value}"
                    if feat_key in self.feature_to_idx:
                        idx = self.feature_to_idx[feat_key]
                        vec[idx] = max(vec[idx], val)

            elif self.mode == RepresentationMode.BOUND:
                if fact.predicate_type == PredicateType.HAS_ATTRIBUTE:
                    # subject_id is e.g. "part_beak_0" -> extract "beak"
                    part_name = fact.subject_id.replace("part_", "").replace("_0", "")
                    feat_key = f"{part_name}:{fact.attribute}={fact.value}"
                    if feat_key in self.feature_to_idx:
                        idx = self.feature_to_idx[feat_key]
                        vec[idx] = max(vec[idx], val)
                elif fact.predicate_type == PredicateType.HAS_PART:
                    part_name = str(fact.value).replace("part_", "").replace("_0", "")
                    feat_key = f"has_part({part_name})"
                    if feat_key in self.feature_to_idx:
                        idx = self.feature_to_idx[feat_key]
                        vec[idx] = max(vec[idx], val)

            elif self.mode == RepresentationMode.RELATIONAL:
                if fact.predicate_type == PredicateType.HAS_ATTRIBUTE:
                    part_name = fact.subject_id.replace("part_", "").replace("_0", "")
                    feat_key = f"{part_name}:{fact.attribute}={fact.value}"
                    if feat_key in self.feature_to_idx:
                        idx = self.feature_to_idx[feat_key]
                        vec[idx] = max(vec[idx], val)
                elif fact.predicate_type == PredicateType.HAS_PART:
                    part_name = str(fact.value).replace("part_", "").replace("_0", "")
                    feat_key = f"has_part({part_name})"
                    if feat_key in self.feature_to_idx:
                        idx = self.feature_to_idx[feat_key]
                        vec[idx] = max(vec[idx], val)
                elif fact.predicate_type == PredicateType.SPATIAL:
                    p1 = fact.subject_id.replace("part_", "").replace("_0", "")
                    p2 = str(fact.value).replace("part_", "").replace("_0", "")
                    feat_key = f"{fact.predicate}({p1}, {p2})"
                    if feat_key in self.feature_to_idx:
                        idx = self.feature_to_idx[feat_key]
                        vec[idx] = max(vec[idx], val)

        return vec

    def compile_batch(
        self, dbs: List[FactDatabase], use_probabilities: bool = False
    ) -> np.ndarray:
        """Compiles a batch of databases into a 2D matrix [N, num_features]."""
        matrix = np.zeros((len(dbs), self.num_features), dtype=np.float32)
        for i, db in enumerate(dbs):
            matrix[i] = self.compile_database(db, use_probabilities=use_probabilities)
        return matrix
