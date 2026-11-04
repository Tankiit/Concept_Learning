"""
NLX Conditioning Adapter
========================
Prepares strictly conditioned prompt contexts for Natural Language Explanation (NLX)
decoders, ensuring that text generation is constrained entirely to verified,
grounded facts and formal certificates to eliminate hallucination.

Every fact handed to a decoder is PRE-VERBALIZED with the deterministic
polarity-aware template, so an LLM decoder receives "no white belly color
(polarity: absent)" rather than a raw feature name it might phrase as
presence.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any

from ..predicates.schema import FactDatabase
from ..explanations.schema import SufficientExplanation
from ..explanations.evidence import VisualEvidenceCard
from .templates import ExplanationTemplateGenerator, is_negated_condition


class NLXConditioningAdapter:
    """
    Constructs bounded decoder prompts and executes grounded decoding.
    """
    def __init__(self, template_style: str = "scientific"):
        self.template_gen = ExplanationTemplateGenerator(style=template_style)

    def build_decoder_prompt(
        self,
        explanation: SufficientExplanation,
        evidence_cards: List[VisualEvidenceCard],
        db: Optional[FactDatabase] = None,
    ) -> Dict[str, Any]:
        """
        Builds a structured prompt containing only certified facts, each with a
        pre-verbalized polarity-correct clause.
        """
        species = explanation.target_class_name
        verified_facts = []

        for card in evidence_cards:
            clause = self.template_gen.verbalize_condition(card)
            polarity = (
                "absent" if is_negated_condition(card.condition_str) else "present"
            )
            item = {
                "clause": clause,
                "polarity": polarity,
                "feature": card.feature_name,
                "part": card.part_name,
                "confidence": card.confidence,
                "bbox": card.bbox,
                "prototype_id": card.prototype_id,
            }
            verified_facts.append(item)

        system_instruction = (
            "You are a scientific visual reasoner explaining fine-grained bird classification. "
            "You MUST strictly condition your explanation on the provided verified visual facts. "
            "Do NOT speculate, assume, or introduce any visual properties that are not explicitly listed. "
            "You MUST preserve the stated polarity of every clause exactly: a fact marked "
            "'absent' must be verbalized as absence (e.g. 'no white belly color'), never as presence."
        )

        user_content = (
            f"Classify Bird As: {species}\n"
            f"Sufficient Conditions: {len(verified_facts)}\n"
            f"Verified Evidence List:\n"
            + "\n".join(
                f"- {f['clause']} [polarity: {f['polarity']}; "
                + (
                    f"confidence: {f['confidence']:.2f}; "
                    if f["confidence"] is not None
                    else "confidence: n/a; "
                )
                + f"region: {f['part']}]"
                for f in verified_facts
            )
            + "\nGenerate a faithful, concise scientific explanation."
        )

        return {
            "system_instruction": system_instruction,
            "user_prompt": user_content,
            "verified_facts": verified_facts,
            "target_species": species,
        }

    def decode(
        self,
        explanation: SufficientExplanation,
        evidence_cards: List[VisualEvidenceCard],
        decoder_type: str = "template",
    ) -> str:
        """
        Executes grounded decoding. Only the deterministic template decoder is
        implemented in this repo (no LLM backend is wired in); requesting any
        other decoder type raises instead of silently falling back — callers
        must never believe a different decoder ran.
        """
        if decoder_type == "template":
            return self.template_gen.generate_sufficient_explanation(
                explanation, evidence_cards
            )
        raise NotImplementedError(
            f"decoder_type={decoder_type!r} is not implemented; only 'template' "
            f"is available (no LLM backend exists in this repo)."
        )
