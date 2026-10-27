"""
Independent Tree Sufficiency Verifier
=====================================
Performs formal verification of explanation sufficiency for decision trees:
Given a set of candidate conditions C and a target prediction y, checks whether
EVERY path in the tree consistent with C terminates in a leaf predicting y.
Includes an exact minimal prime implicant extraction algorithm.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..reasoning.tree_head import CertifiableTreeHead, DecisionPathStep
from .schema import (
    ExplanationCondition,
    SufficientExplanation,
    VerificationCertificate,
    VerificationStatus,
)


class TreeSufficiencyVerifier:
    """
    Independent mathematical sufficiency verifier for Decision Trees.
    """
    def __init__(self, clf: DecisionTreeClassifier, feature_names: List[str]):
        self.clf = clf
        self.tree = clf.tree_
        self.feature_names = feature_names
        self.num_features = self.tree.n_features
        self._cache_leaves()

    def _cache_leaves(self) -> None:
        """Extracts and caches the branch constraints and predictions of all leaves."""
        self.leaves: List[Dict[str, Any]] = []

        def recurse(node_id: int, current_path: List[Tuple[int, str, float]]):
            # Check if leaf node
            is_leaf = (
                self.tree.children_left[node_id] == -1
                or self.tree.feature[node_id] < 0
            )
            if is_leaf:
                counts = self.tree.value[node_id][0]
                pred_class = int(np.argmax(counts))
                self.leaves.append({
                    "node_id": node_id,
                    "pred_class": pred_class,
                    "path": list(current_path),
                })
            else:
                f_idx = int(self.tree.feature[node_id])
                thresh = float(self.tree.threshold[node_id])

                # Left branch: feature <= thresh
                recurse(
                    self.tree.children_left[node_id],
                    current_path + [(f_idx, "<=", thresh)],
                )
                # Right branch: feature > thresh
                recurse(
                    self.tree.children_right[node_id],
                    current_path + [(f_idx, ">", thresh)],
                )

        recurse(0, [])

    @staticmethod
    def _are_constraints_compatible(
        constraints: List[Tuple[int, str, float]]
    ) -> bool:
        """
        Checks whether a set of (feature_idx, op, thresh) constraints is mutually satisfiable.
        """
        lower_bounds: Dict[int, float] = {}
        upper_bounds: Dict[int, float] = {}

        for f_idx, op, thresh in constraints:
            if op == "<=":
                if f_idx in upper_bounds:
                    upper_bounds[f_idx] = min(upper_bounds[f_idx], thresh)
                else:
                    upper_bounds[f_idx] = thresh
            elif op == ">":
                if f_idx in lower_bounds:
                    lower_bounds[f_idx] = max(lower_bounds[f_idx], thresh)
                else:
                    lower_bounds[f_idx] = thresh

            # Contradiction check: lower bound must be strictly less than upper bound
            if f_idx in lower_bounds and f_idx in upper_bounds:
                if lower_bounds[f_idx] >= upper_bounds[f_idx]:
                    return False

        return True

    def verify_sufficiency(
        self, conditions: List[ExplanationCondition], target_class: int
    ) -> VerificationCertificate:
        """
        Verifies whether conditions are formally sufficient for target_class.
        Checks if ALL tree leaves consistent with conditions predict target_class.
        """
        cond_constraints: List[Tuple[int, str, float]] = [
            (c.feature_idx, c.operator, c.threshold) for c in conditions
        ]

        # 1. Contradiction within condition set itself
        if not self._are_constraints_compatible(cond_constraints):
            return VerificationCertificate(
                status=VerificationStatus.FALSIFIED,
                target_class=target_class,
                total_leaves=len(self.leaves),
                consistent_leaves=0,
                proof_details={"error": "Self-contradictory condition set"},
            )

        # 2. Check each leaf for consistency with conditions
        consistent_count = 0
        conflicting_leaves = []

        for leaf in self.leaves:
            combined = cond_constraints + leaf["path"]
            if self._are_constraints_compatible(combined):
                consistent_count += 1
                if leaf["pred_class"] != target_class:
                    conflicting_leaves.append({
                        "node_id": leaf["node_id"],
                        "leaf_class": leaf["pred_class"],
                    })

        # Sufficient if at least one leaf is reached and ZERO conflicting leaves are reachable
        is_sufficient = (consistent_count > 0) and (len(conflicting_leaves) == 0)
        status = VerificationStatus.VERIFIED if is_sufficient else VerificationStatus.FALSIFIED

        return VerificationCertificate(
            status=status,
            target_class=target_class,
            total_leaves=len(self.leaves),
            consistent_leaves=consistent_count,
            proof_details={"conflicting_leaves": conflicting_leaves},
        )


def extract_minimal_sufficient_conditions(
    tree_head: CertifiableTreeHead,
    x: np.ndarray,
    target_class: int,
    sample_id: int = 0,
) -> SufficientExplanation:
    """
    Computes the minimal prime implicant sufficient explanation for instance x.
    Uses backward elimination pruning over the verified decision path.
    """
    assert tree_head.is_fitted
    verifier = TreeSufficiencyVerifier(tree_head.clf, tree_head.feature_names)

    # 1. Start from full path steps
    path_steps: List[DecisionPathStep] = tree_head.get_decision_path(x)
    initial_conditions = [
        ExplanationCondition(
            feature_idx=step.feature_idx,
            feature_name=step.feature_name,
            operator="<=" if step.is_left else ">",
            threshold=step.threshold,
            observed_value=step.value,
        )
        for step in path_steps
    ]

    class_name = (
        tree_head.class_names[target_class]
        if target_class < len(tree_head.class_names)
        else str(target_class)
    )

    # Verify initial path sufficiency
    initial_cert = verifier.verify_sufficiency(initial_conditions, target_class)
    if initial_cert.status != VerificationStatus.VERIFIED:
        return SufficientExplanation(
            sample_id=sample_id,
            target_class=target_class,
            target_class_name=class_name,
            conditions=initial_conditions,
            certificate=initial_cert,
        )

    # 2. Backward elimination to find minimal subset (Prime Implicant)
    minimal_conditions = list(initial_conditions)
    for candidate in initial_conditions:
        trial_conditions = [c for c in minimal_conditions if c != candidate]
        # An empty set of conditions cannot be sufficient unless entire tree is a single constant class
        if not trial_conditions and len(verifier.leaves) > 1:
            continue
        cert = verifier.verify_sufficiency(trial_conditions, target_class)
        if cert.status == VerificationStatus.VERIFIED:
            minimal_conditions = trial_conditions

    # Final certificate on the minimal subset
    final_cert = verifier.verify_sufficiency(minimal_conditions, target_class)

    return SufficientExplanation(
        sample_id=sample_id,
        target_class=target_class,
        target_class_name=class_name,
        conditions=minimal_conditions,
        certificate=final_cert,
    )
