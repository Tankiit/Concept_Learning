"""
Explanation Faithfulness Auditor
================================
Audits generated natural language explanations against formal conditions to detect:
1. Omitted claims: Verified conditions absent from the verbalized text.
2. Unsupported claims / Hallucinations: Visual attributes claimed in the text
   that are NOT backed by certified conditions or grounded facts.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import re

from ..explanations.schema import SufficientExplanation, ExplanationCondition
from ..predicates.schema import FactDatabase


@dataclass
class AuditReport:
    """Detailed audit report quantifying natural language explanation faithfulness."""
    is_faithful: bool
    omitted_conditions: List[str]
    unsupported_claims: List[str]
    faithfulness_precision: float
    faithfulness_recall: float
    audit_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_faithful": self.is_faithful,
            "omitted_conditions": self.omitted_conditions,
            "unsupported_claims": self.unsupported_claims,
            "faithfulness_precision": float(self.faithfulness_precision),
            "faithfulness_recall": float(self.faithfulness_recall),
            "audit_summary": self.audit_summary,
        }


class ExplanationAuditor:
    """
    Auditor that scans natural language explanations against grounded conditions.
    """
    def __init__(self):
        # Known vocabulary tokens for parts and attributes
        self.known_colors = {"yellow", "black", "white", "red", "brown", "blue", "grey", "gray"}
        self.known_shapes = {"curved", "cone", "dagger", "hooked", "needle", "forked"}
        self.known_parts = {
            "beak", "bill", "crown", "forehead", "breast", "belly", "wing", "tail", "throat", "back"
        }

    def _normalize_text(self, text: str) -> str:
        return text.lower().replace("-", " ").replace("_", " ")

    def audit(
        self,
        explanation_text: str,
        sufficient_explanation: SufficientExplanation,
        db: Optional[FactDatabase] = None,
    ) -> AuditReport:
        """
        Runs an audit of explanation_text against certified conditions.
        """
        norm_text = self._normalize_text(explanation_text)
        omitted: List[str] = []

        # 1. Check for omitted conditions
        for cond in sufficient_explanation.conditions:
            f_name = cond.feature_name.lower().replace("_", " ")
            # Check if key tokens in feature name are present in text
            if ":" in f_name and "=" in f_name:
                part, attr_val = f_name.split(":", 1)
                attr, val = attr_val.split("=", 1)
                # Should mention part or attribute/value
                part_mentioned = (part in norm_text) or (part == "beak" and "bill" in norm_text)
                val_mentioned = val in norm_text
                if not (part_mentioned and val_mentioned):
                    omitted.append(cond.feature_name)
            elif "=" in f_name:
                attr, val = f_name.split("=", 1)
                if val not in norm_text:
                    omitted.append(cond.feature_name)

        # 2. Check for unsupported claims / hallucinations
        # Extract visual tokens in text that aren't in verified conditions
        allowed_tokens: Set[str] = set()
        for cond in sufficient_explanation.conditions:
            for tok in re.split(r"[:=_\s\(\),]+", cond.feature_name.lower()):
                if tok:
                    allowed_tokens.add(tok)
        if "beak" in allowed_tokens:
            allowed_tokens.add("bill")

        unsupported: List[str] = []
        words = set(re.findall(r"\b[a-z]+\b", norm_text))

        # Check color tokens mentioned
        for color in self.known_colors:
            if color in words and color not in allowed_tokens:
                unsupported.append(f"color:{color}")

        # Check shape tokens mentioned
        for shape in self.known_shapes:
            if shape in words and shape not in allowed_tokens:
                unsupported.append(f"shape:{shape}")

        total_conds = len(sufficient_explanation.conditions)
        covered_conds = total_conds - len(omitted)
        recall = (covered_conds / total_conds) if total_conds > 0 else 1.0

        n_claims = covered_conds + len(unsupported)
        precision = (covered_conds / n_claims) if n_claims > 0 else 1.0

        is_faithful = (len(omitted) == 0) and (len(unsupported) == 0)

        summary = (
            f"Audit {'PASSED' if is_faithful else 'FAILED'}: "
            f"{len(omitted)} omitted conditions, {len(unsupported)} unsupported claims. "
            f"Precision: {precision:.2f}, Recall: {recall:.2f}"
        )

        return AuditReport(
            is_faithful=is_faithful,
            omitted_conditions=omitted,
            unsupported_claims=unsupported,
            faithfulness_precision=precision,
            faithfulness_recall=recall,
            audit_summary=summary,
        )
