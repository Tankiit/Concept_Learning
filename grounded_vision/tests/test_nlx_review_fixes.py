"""
Unit Tests: NLX-layer review fixes (second external review, 2026-11-04)
=======================================================================
Regression tests for the language-layer defects found in collaborator
review of templates.py / nlx_adapter.py / make_activation_nlx.py:

1. LLM decoder prompts must carry polarity (pre-verbalized clause +
   explicit polarity field), or an LLM turns a certified absence into
   presence.
2. decode() must not silently fall back to the template when asked for
   another decoder.
3. The "mathematically sufficient" claim is gated on the certificate
   status being VERIFIED.
4. Evidence cards must match the certified conditions one-to-one.
5. Contrastive text verbalizes TARGET conditions with observed values,
   instead of raw condition strings.
6. Absence clauses report the complement of the reader probability;
   confidence 0.0 still renders; part-prefixed attributes don't repeat
   the part; the empty-reason wording matches tree semantics.
"""

import importlib.util
from pathlib import Path

import pytest

from grounded_vision.explanations.evidence import VisualEvidenceCard
from grounded_vision.explanations.schema import (
    ContrastiveExplanation,
    ExplanationCondition,
    SufficientExplanation,
    VerificationCertificate,
    VerificationStatus,
)
from grounded_vision.language.nlx_adapter import NLXConditioningAdapter
from grounded_vision.language.templates import ExplanationTemplateGenerator


def condition(name, op, thr=0.5, obs=1.0, idx=0):
    return ExplanationCondition(
        feature_idx=idx, feature_name=name, operator=op,
        threshold=thr, observed_value=obs,
    )


def card(name, cond_str, conf=1.0):
    return VisualEvidenceCard(
        condition_str=cond_str, feature_name=name,
        part_name=None, bbox=None, confidence=conf,
    )


def explanation(conds, status=VerificationStatus.VERIFIED, target="Species X"):
    return SufficientExplanation(
        sample_id=0, target_class=0, target_class_name=target,
        conditions=conds,
        certificate=VerificationCertificate(
            status=status, target_class=0, total_leaves=4, consistent_leaves=4,
        ),
    )


def matched(conds, conf=1.0):
    return [card(c.feature_name, c.condition_str, conf) for c in conds]


# --------------------------------------------------------------- adapter ----
def test_decoder_prompt_carries_polarity_and_clause():
    conds = [
        condition("belly:belly_color=white", "<=", obs=0.1, idx=0),
        condition("beak:bill_color=black", ">", obs=0.9, idx=1),
    ]
    adapter = NLXConditioningAdapter()
    prompt = adapter.build_decoder_prompt(explanation(conds), matched(conds))

    neg, pos = prompt["verified_facts"]
    assert neg["polarity"] == "absent"
    assert neg["clause"].startswith("no white belly color")
    assert pos["polarity"] == "present"
    assert "a black bill color" in pos["clause"]
    assert "no white belly color" in prompt["user_prompt"]
    assert "polarity: absent" in prompt["user_prompt"]
    assert "polarity" in prompt["system_instruction"]


def test_decode_rejects_unknown_decoder_type():
    conds = [condition("belly:belly_color=white", "<=")]
    adapter = NLXConditioningAdapter()
    text = adapter.decode(explanation(conds), matched(conds), decoder_type="template")
    assert "classifies" in text
    with pytest.raises(NotImplementedError):
        adapter.decode(explanation(conds), matched(conds), decoder_type="llm")


# -------------------------------------------------------------- templates ----
def test_sufficient_claim_requires_verified_certificate():
    gen = ExplanationTemplateGenerator()
    conds = [condition("beak:bill_color=black", ">", obs=0.9)]

    verified = gen.generate_sufficient_explanation(
        explanation(conds, status=VerificationStatus.VERIFIED), matched(conds)
    )
    assert "mathematically" in verified

    for status in (VerificationStatus.FALSIFIED, VerificationStatus.UNKNOWN):
        text = gen.generate_sufficient_explanation(
            explanation(conds, status=status), matched(conds)
        )
        assert "mathematically" not in text
        assert status.value in text
        assert "NOT guaranteed" in text


def test_card_condition_mismatch_raises():
    gen = ExplanationTemplateGenerator()
    conds = [condition("beak:bill_color=black", ">"), condition("belly:belly_color=white", "<=")]

    with pytest.raises(ValueError):
        gen.generate_sufficient_explanation(explanation(conds), matched(conds)[:1])

    wrong = [matched(conds)[0], card("belly:belly_color=yellow", "x <= 0.5")]
    with pytest.raises(ValueError):
        gen.generate_sufficient_explanation(explanation(conds), wrong)


def test_negated_clause_reports_absence_confidence():
    gen = ExplanationTemplateGenerator()
    neg = gen.verbalize_condition(
        card("belly:belly_color=yellow", "belly:belly_color=yellow <= 0.500", conf=0.12)
    )
    # 0.12 is P(yellow present); the absence clause must report 1 - 0.12
    assert "no yellow belly color [confidence: 0.88]" in neg


def test_zero_confidence_still_renders():
    gen = ExplanationTemplateGenerator()
    pos = gen.verbalize_condition(
        card("beak:bill_color=black", "beak:bill_color=black > 0.500", conf=0.0)
    )
    assert "[confidence: 0.00]" in pos


def test_part_prefixed_attribute_drops_redundant_part():
    gen = ExplanationTemplateGenerator()
    redundant = gen.verbalize_condition(
        card("belly:belly_color=white", "belly:belly_color=white <= 0.500", conf=0.25)
    )
    assert redundant == "no white belly color [confidence: 0.75]"
    assert "on the belly" not in redundant

    distinct = gen.verbalize_condition(
        card("beak:bill_color=black", "beak:bill_color=black > 0.500")
    )
    assert "a black bill color on the beak" in distinct


def test_empty_reason_set_wording():
    gen = ExplanationTemplateGenerator()
    text = gen.generate_sufficient_explanation(explanation([]), [])
    # An empty prime implicant means the head predicts this class everywhere.
    assert "for every input" in text
    assert "prior distribution" not in text


def test_contrastive_verbalizes_target_conditions():
    """conditions_to_alter holds TARGET conditions in the contrast leaf's
    required direction, with the currently observed (violating) value."""
    gen = ExplanationTemplateGenerator()
    contrast = ContrastiveExplanation(
        sample_id=0, original_class=0, contrast_class=1,
        conditions_to_alter=[
            condition("breast:breast_color=yellow", ">", thr=0.5, obs=0.2, idx=0),
            condition("belly:belly_color=yellow", "<=", thr=0.5, obs=0.9, idx=1),
        ],
    )
    text = gen.generate_contrastive_explanation(contrast, "Species Y")
    assert "a yellow breast color (currently 0.20; requires > 0.500)" in text
    assert "no yellow belly color (currently 0.90; requires <= 0.500)" in text
    assert "would become Species Y" in text
    # no reader confidence attaches to a hypothetical target state
    assert "confidence" not in text


# ----------------------------------------------------- make_activation_nlx ----
def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_activation_nlx.py"
    spec = importlib.util.spec_from_file_location("make_activation_nlx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_activation_nlx_guards_empty_and_missing_grounding():
    mod = _load_script()
    empty = mod.build_nlx({"prediction": 2, "grounding": []})
    assert "because ." not in empty["nlx"]["text"]
    assert "no grounded conditions recorded" in empty["nlx"]["text"]

    missing = mod.build_nlx({"prediction": 2})
    assert "no grounded conditions recorded" in missing["nlx"]["text"]

    grounded = mod.build_nlx({
        "prediction": 2,
        "grounding": [{
            "activation_channel": 39,
            "tile": {"row": 1, "col": 2},
            "condition": {"operator": ">", "threshold": 1.61, "observed_value": 2.0},
        }],
    })
    assert "channel 39" in grounded["nlx"]["text"]
    assert "row 2, column 3" in grounded["nlx"]["text"]
    # faithfulness/abstention claims are measured by the auditor, not asserted
    assert "faithfulness" not in grounded["nlx"]
    assert "semantic_abstention" not in grounded["nlx"]
