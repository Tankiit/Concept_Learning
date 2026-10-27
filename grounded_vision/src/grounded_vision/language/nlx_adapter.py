"""
NLX Conditioning Adapter
========================
Prepares strictly conditioned prompt contexts for Natural Language Explanation (NLX)
decoders, ensuring that text generation is constrained entirely to verified,
grounded facts and formal certificates to eliminate hallucination.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any

from ..predicates.schema import FactDatabase
from ..explanations.schema import SufficientExplanation
from ..explanations.evidence import VisualEvidenceCard
from .templates import ExplanationTemplateGenerator


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
        Builds a structured prompt containing only certified facts.
        """
        species = explanation.target_class_name
        verified_facts = []

        for card in evidence_cards:
            item = {
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
            "Do NOT speculate, assume, or introduce any visual properties that are not explicitly listed."
        )

        user_content = (
            f"Classify Bird As: {species}\n"
            f"Sufficient Conditions: {len(verified_facts)}\n"
            f"Verified Evidence List:\n"
            + "\n".join(
                f"- {f['feature']} (confidence: {f['confidence']:.2f}, region: {f['part']})"
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
        Executes grounded decoding. Defaults to faithful deterministic template generation.
        """
        if decoder_type == "template":
            return self.template_gen.generate_sufficient_explanation(
                explanation, evidence_cards
            )
        else:
            # Fallback to deterministic template
            return self.template_gen.generate_sufficient_explanation(
                explanation, evidence_cards
            )
