"""
PyXAI Backend and Native Abductive Solver
=========================================
Interfaces with PyXAI (Python eXplainable AI) for tree-based formal explanations.
Gracefully provides native fallback to the exact TreeSufficiencyVerifier prime-implicant
solver when PyXAI is not installed or platform binaries are incompatible.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..reasoning.tree_head import CertifiableTreeHead
from .schema import (
    ExplanationCondition,
    SufficientExplanation,
    ContrastiveExplanation,
    VerificationCertificate,
    VerificationStatus,
)
from .tree_verifier import TreeSufficiencyVerifier, extract_minimal_sufficient_conditions


class PyXAIBackend:
    """
    Unified manager for formal tree explanation generation.

    HONESTY NOTE: despite the name, PyXAI is NOT used to compute anything
    here. `pyxai_available` records whether the PyXAI module can be imported
    (its bundled d4 compiler is an x86-64 Linux binary and cannot run on
    macOS). All sufficient/contrastive explanations are computed by the
    native exact solver (TreeSufficiencyVerifier). The actual PyXAI
    integration — independent certification and minimum-cardinality audits
    on Linux — lives in scripts/run_pyxai_modal.py.
    """

    #: Identifies the engine that produces explanations from this backend.
    solver = "native_tree_verifier"

    def __init__(self, tree_head: CertifiableTreeHead):
        self.tree_head = tree_head
        self.pyxai_available = False
        self.pyxai_explainer = None

        self._init_pyxai()

    def _init_pyxai(self) -> None:
        """Attempts to load PyXAI module and prepare explainer."""
        try:
            import pyxai  # type: ignore
            # PyXAI requires dataset / model builder
            # If pyxai is available, initialize
            self.pyxai_available = True
        except (ImportError, Exception):
            self.pyxai_available = False

    def explain_sufficient(
        self, x: np.ndarray, target_class: int, sample_id: int = 0
    ) -> SufficientExplanation:
        """
        Generates a certified minimal sufficient explanation (prime implicant).
        """
        # Always verified by independent TreeSufficiencyVerifier
        return extract_minimal_sufficient_conditions(
            self.tree_head, x, target_class, sample_id=sample_id
        )

    def explain_contrastive(
        self,
        x: np.ndarray,
        original_class: int,
        contrast_class: Optional[int] = None,
        sample_id: int = 0,
    ) -> Optional[ContrastiveExplanation]:
        """
        Finds the minimal set of conditions whose alteration would change the prediction
        from original_class to contrast_class.
        """
        assert self.tree_head.is_fitted
        rules = self.tree_head.export_rules()

        # Find leaves predicting contrast_class
        contrast_leaves = [
            r for r in rules
            if (contrast_class is None and r["predicted_class"] != original_class)
            or (contrast_class is not None and r["predicted_class"] == contrast_class)
        ]

        if not contrast_leaves:
            return None

        # Compare instance x to each contrast leaf condition set
        best_alterations: Optional[List[ExplanationCondition]] = None
        chosen_contrast_class = original_class

        for leaf in contrast_leaves:
            alterations: List[ExplanationCondition] = []
            for cond in leaf["conditions"]:
                f_idx = cond["feature_idx"]
                thresh = cond["threshold"]
                is_left = cond["is_left"]
                val = float(x[f_idx])

                # Check if x violates this condition
                violated = (val > thresh) if is_left else (val <= thresh)
                if violated:
                    alterations.append(
                        ExplanationCondition(
                            feature_idx=f_idx,
                            feature_name=cond["feature_name"],
                            operator="<=" if is_left else ">",
                            threshold=thresh,
                            observed_value=val,
                        )
                    )

            if best_alterations is None or len(alterations) < len(best_alterations):
                best_alterations = alterations
                chosen_contrast_class = leaf["predicted_class"]

        if best_alterations is None:
            return None

        return ContrastiveExplanation(
            sample_id=sample_id,
            original_class=original_class,
            contrast_class=chosen_contrast_class,
            conditions_to_alter=best_alterations,
        )
