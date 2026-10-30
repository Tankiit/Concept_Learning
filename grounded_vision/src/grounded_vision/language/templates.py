"""
Deterministic Verbalization Templates
=====================================
Transforms formal explanation conditions and visual evidence cards into
rigorous, grammatically fluent, and audit-ready scientific explanations.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any

from ..explanations.schema import SufficientExplanation, ContrastiveExplanation
from ..explanations.evidence import VisualEvidenceCard


class ExplanationTemplateGenerator:
    """
    Generates deterministic natural language explanations from certified conditions.
    """
    def __init__(self, style: str = "scientific"):
        self.style = style

    def verbalize_condition(self, card: VisualEvidenceCard) -> str:
        """Verbalizes a single visual evidence condition.

        Polarity matters: a certified condition like
        "belly:belly_color=white <= 0.500" certifies the ABSENCE of the
        concept, so it must be verbalized as absence ("no white belly color"),
        never as presence.
        """
        f_name = card.feature_name
        conf_str = f" [confidence: {card.confidence:.2f}]" if card.confidence else ""
        negated = "<=" in card.condition_str

        if ":" in f_name and "=" in f_name:
            part, attr_val = f_name.split(":", 1)
            attr, val = attr_val.split("=", 1)
            attr_clean = attr.replace("_", " ")
            if negated:
                return f"no {val} {attr_clean} on the {part}{conf_str}"
            return f"a {val} {attr_clean} on the {part}{conf_str}"
        elif "has_part" in f_name:
            part = f_name.replace("has_part(", "").replace(")", "").strip()
            if negated:
                return f"the absence of the {part}{conf_str}"
            return f"the presence of the {part}{conf_str}"
        elif "=" in f_name:
            attr, val = f_name.split("=", 1)
            attr_clean = attr.replace("_", " ")
            if negated:
                return f"no {val} {attr_clean}{conf_str}"
            return f"a {val} {attr_clean}{conf_str}"
        elif "(" in f_name:
            # Spatial/relational e.g. above(crown, beak)
            rel = f_name.split("(")[0]
            args = f_name.split("(")[1].replace(")", "").split(",")
            p1 = args[0].strip()
            p2 = args[1].strip() if len(args) > 1 else "other part"
            return f"the {p1} situated {rel.replace('_', ' ')} the {p2}{conf_str}"
        return f"{f_name}{conf_str}"

    def generate_sufficient_explanation(
        self,
        explanation: SufficientExplanation,
        evidence_cards: List[VisualEvidenceCard],
    ) -> str:
        """Generates natural language text for a certified sufficient explanation."""
        species = explanation.target_class_name
        n_conds = len(explanation.conditions)

        if n_conds == 0:
            return f"The model classifies this sample as {species} based on prior distribution alone."

        cond_texts = [
            f"({i+1}) {self.verbalize_condition(card)}"
            for i, card in enumerate(evidence_cards)
        ]
        conds_joined = ", ".join(cond_texts)

        if self.style == "scientific":
            return (
                f"The model classifies this bird as {species} because it observed the following certified "
                f"sufficient conditions: {conds_joined}. These {n_conds} conditions are mathematically "
                f"sufficient under the deployed decision rules to guarantee this prediction."
            )
        else:
            return (
                f"This bird is recognized as {species} because: {conds_joined}."
            )

    def generate_contrastive_explanation(
        self,
        contrast: ContrastiveExplanation,
        contrast_class_name: str,
    ) -> str:
        """Generates natural language text for a contrastive explanation."""
        cond_strs = [c.condition_str for c in contrast.conditions_to_alter]
        joined = ", ".join(cond_strs)
        return (
            f"The model did not classify this bird as {contrast_class_name} because the following "
            f"required properties were not satisfied: {joined}."
        )


def verbalize_sufficient_explanation(
    explanation: SufficientExplanation,
    evidence_cards: List[VisualEvidenceCard],
    style: str = "scientific",
) -> str:
    """Helper function to quickly obtain natural language verbalization."""
    gen = ExplanationTemplateGenerator(style=style)
    return gen.generate_sufficient_explanation(explanation, evidence_cards)
