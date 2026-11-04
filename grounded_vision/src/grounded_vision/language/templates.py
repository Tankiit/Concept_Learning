"""
Deterministic Verbalization Templates
=====================================
Transforms formal explanation conditions and visual evidence cards into
rigorous, grammatically fluent, and audit-ready scientific explanations.

Faithfulness scope: the text is faithful to the MODEL'S COMPUTATION —
each clause restates a certified condition over predicted concept values.
If a concept reader misreads the image, the sentence remains faithful to
the model and wrong about the bird. Claims below never assert image truth.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any

from ..explanations.schema import (
    SufficientExplanation,
    ContrastiveExplanation,
    ExplanationCondition,
    VerificationStatus,
)
from ..explanations.evidence import VisualEvidenceCard


def is_negated_condition(condition_str: str) -> bool:
    """Single source of truth for parsing polarity out of a condition string
    of the schema-controlled form ``{feature_name} {operator} {threshold}``.
    Prefer passing the condition object itself when available."""
    return "<=" in condition_str


class ExplanationTemplateGenerator:
    """
    Generates deterministic natural language explanations from certified conditions.
    """
    def __init__(self, style: str = "scientific"):
        self.style = style

    def verbalize_condition(
        self,
        card: VisualEvidenceCard,
        condition: Optional[ExplanationCondition] = None,
    ) -> str:
        """Verbalizes a single visual evidence condition.

        Polarity matters: a certified condition like
        "belly:belly_color=white <= 0.500" certifies the ABSENCE of the
        concept, so it must verbalize as absence ("no white belly color"),
        never as presence. Polarity is taken from the condition's operator
        when the condition object is supplied, and parsed from the card's
        condition string otherwise.

        Confidence polarity matters too: ``card.confidence`` is the reader's
        probability that the attribute IS present, so an absence clause
        reports the complement (1 - p) — "no yellow belly color
        [confidence: 0.88]" must mean 88% sure it is absent.
        """
        f_name = card.feature_name
        if condition is not None:
            negated = condition.operator == "<="
        else:
            negated = is_negated_condition(card.condition_str)

        conf_str = ""
        if card.confidence is not None:
            conf_val = (1.0 - card.confidence) if negated else card.confidence
            conf_str = f" [confidence: {conf_val:.2f}]"

        if ":" in f_name and "=" in f_name:
            part, attr_val = f_name.split(":", 1)
            attr, val = attr_val.split("=", 1)
            attr_clean = attr.replace("_", " ")
            # "belly:belly_color" -> "belly color": the attribute name already
            # carries the part; "no white belly color on the belly" is redundant.
            if attr_clean.startswith(part):
                if negated:
                    return f"no {val} {attr_clean}{conf_str}"
                return f"a {val} {attr_clean}{conf_str}"
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
            # Spatial/relational e.g. above(crown, beak).
            # A certified condition like "above(crown, beak) <= 0.500"
            # certifies the ABSENCE of the relation and must verbalize as
            # absence, exactly like attribute conditions.
            rel = f_name.split("(")[0]
            args = f_name.split("(")[1].replace(")", "").split(",")
            p1 = args[0].strip()
            p2 = args[1].strip() if len(args) > 1 else "other part"
            negation = "not " if negated else ""
            return f"the {p1} {negation}situated {rel.replace('_', ' ')} the {p2}{conf_str}"
        return f"{f_name}{conf_str}"

    def generate_sufficient_explanation(
        self,
        explanation: SufficientExplanation,
        evidence_cards: List[VisualEvidenceCard],
    ) -> str:
        """Generates natural language text for a certified sufficient explanation.

        The mathematical-sufficiency claim is made ONLY when the attached
        certificate's status is VERIFIED; any other status gets wording that
        names it. Evidence cards must correspond one-to-one, in order, to the
        certified conditions — a dropped or reordered card raises rather than
        silently producing a sentence whose count and clauses disagree.
        """
        species = explanation.target_class_name
        conditions = explanation.conditions
        n_conds = len(conditions)

        if len(evidence_cards) != n_conds:
            raise ValueError(
                f"evidence cards ({len(evidence_cards)}) do not match certified "
                f"conditions ({n_conds}); refusing to verbalize a mismatched set"
            )
        for card, cond in zip(evidence_cards, conditions):
            if card.feature_name != cond.feature_name:
                raise ValueError(
                    f"evidence card feature '{card.feature_name}' does not match "
                    f"condition '{cond.feature_name}' at the same position"
                )

        if n_conds == 0:
            # An empty prime implicant means no feature assignment can change
            # the outcome: the head predicts this class for EVERY input.
            return (
                f"The model predicts {species} for every input under the deployed "
                f"decision rules (the sufficient condition set is empty)."
            )

        cond_texts = [
            f"({i+1}) {self.verbalize_condition(card, cond)}"
            for i, (card, cond) in enumerate(zip(evidence_cards, conditions))
        ]
        conds_joined = ", ".join(cond_texts)

        verified = (
            explanation.certificate is not None
            and explanation.certificate.status == VerificationStatus.VERIFIED
        )
        if verified:
            closing = (
                f"These {n_conds} conditions are mathematically sufficient, "
                f"given the predicted concept values, to guarantee this "
                f"prediction under the deployed decision rules."
            )
        else:
            status = (
                explanation.certificate.status.value
                if explanation.certificate is not None
                else "missing"
            )
            closing = (
                f"These {n_conds} conditions are a candidate sufficient set, "
                f"given the predicted concept values, but the verification "
                f"certificate status is '{status}' — mathematical sufficiency "
                f"is NOT guaranteed for this prediction."
            )

        if self.style == "scientific":
            return (
                f"The model classifies this bird as {species} because it observed the following certified "
                f"sufficient conditions: {conds_joined}. {closing}"
            )
        else:
            return (
                f"This bird is recognized as {species} because: {conds_joined}. {closing}"
            )

    def generate_contrastive_explanation(
        self,
        contrast: ContrastiveExplanation,
        contrast_class_name: str,
    ) -> str:
        """Generates natural language text for a contrastive explanation.

        Semantics of ``conditions_to_alter`` (see ``PyXAIBackend.explain_contrastive``):
        each entry is a TARGET condition required by the contrast leaf, stated
        in the leaf's required direction (its operator), carrying the currently
        observed (violating) value. So the text says what would need to hold,
        with the current value attached — for both ``<=`` and ``>`` targets.
        """
        clauses = []
        for cond in contrast.conditions_to_alter:
            # No reader confidence attaches to a hypothetical target state;
            # confidence=None suppresses the confidence tag.
            card = VisualEvidenceCard(
                condition_str=cond.condition_str,
                feature_name=cond.feature_name,
                part_name=None,
                bbox=None,
                confidence=None,
            )
            phrase = self.verbalize_condition(card, cond)
            clauses.append(
                f"{phrase} (currently {cond.observed_value:.2f}; "
                f"requires {cond.operator} {cond.threshold:.3f})"
            )
        joined = ", ".join(clauses)
        return (
            f"The model did not classify this bird as {contrast_class_name} because the following "
            f"properties do not currently hold: {joined}. If they did, the prediction "
            f"would become {contrast_class_name}."
        )


def verbalize_sufficient_explanation(
    explanation: SufficientExplanation,
    evidence_cards: List[VisualEvidenceCard],
    style: str = "scientific",
) -> str:
    """Helper function to quickly obtain natural language verbalization."""
    gen = ExplanationTemplateGenerator(style=style)
    return gen.generate_sufficient_explanation(explanation, evidence_cards)
