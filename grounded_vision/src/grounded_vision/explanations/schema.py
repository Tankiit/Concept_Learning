"""
Explanation Schema and Verification Scope
=========================================
Formal definitions of explanation components:
1. ExplanationCondition: A single literal constraint (feature op threshold).
2. SufficientExplanation: Prime implicant condition set guaranteeing prediction.
3. ContrastiveExplanation: Minimal alteration set leading to alternative prediction.
4. VerificationCertificate: Proof of mathematical sufficiency.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"


@dataclass
class ExplanationCondition:
    """A single conditional literal in an explanation."""
    feature_idx: int
    feature_name: str
    operator: str  # "<=" or ">"
    threshold: float
    observed_value: float

    @property
    def condition_str(self) -> str:
        return f"{self.feature_name} {self.operator} {self.threshold:.3f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "feature_name": self.feature_name,
            "operator": self.operator,
            "threshold": float(self.threshold),
            "observed_value": float(self.observed_value),
            "condition_str": self.condition_str,
        }


@dataclass
class VerificationCertificate:
    """Mathematical audit certificate verifying explanation correctness."""
    status: VerificationStatus
    target_class: int
    total_leaves: int
    consistent_leaves: int
    verifier_name: str = "TreeSufficiencyVerifier"
    proof_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SufficientExplanation:
    """
    Minimal sufficient explanation (Abductive explanation / Prime implicant).
    If all conditions in this set hold, the model is guaranteed to predict target_class.
    """
    sample_id: int
    target_class: int
    target_class_name: str
    conditions: List[ExplanationCondition]
    certificate: VerificationCertificate
    sparsity: int = field(init=False)

    def __post_init__(self):
        self.sparsity = len(self.conditions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "target_class": self.target_class,
            "target_class_name": self.target_class_name,
            "sparsity": self.sparsity,
            "conditions": [c.to_dict() for c in self.conditions],
            "certificate": {
                "status": self.certificate.status.value,
                "verifier": self.certificate.verifier_name,
                "total_leaves": self.certificate.total_leaves,
                "consistent_leaves": self.certificate.consistent_leaves,
            },
        }


@dataclass
class ContrastiveExplanation:
    """
    Contrastive explanation ("Why P rather than Q?"):
    Specifies minimal condition modifications that would shift prediction to contrast_class.
    """
    sample_id: int
    original_class: int
    contrast_class: int
    conditions_to_alter: List[ExplanationCondition]


@dataclass
class VerifiedExplanation:
    """Complete verified explanation container linking logic to perception evidence."""
    sufficient_explanation: SufficientExplanation
    contrastive_explanation: Optional[ContrastiveExplanation] = None
    evidence_cards: List[Dict[str, Any]] = field(default_factory=list)
    natural_language: Optional[str] = None
