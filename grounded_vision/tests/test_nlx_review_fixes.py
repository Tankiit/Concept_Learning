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


# ------------------------------------------------- round 2: auditor binding ----
def test_auditor_catches_polarity_flip():
    """'a white belly color' is a polarity flip of a certified
    belly:belly_color=white <= 0.5 — it must fail the audit."""
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    conds = [condition("belly:belly_color=white", "<=", obs=0.1)]
    exp = explanation(conds)
    gen = ExplanationTemplateGenerator()
    good = gen.generate_sufficient_explanation(exp, matched(conds))
    assert auditor.audit(good, exp).is_faithful

    flipped = (
        "The model classifies this bird as Species X because it observed "
        "(1) a white belly color. These 1 conditions are mathematically "
        "sufficient."
    )
    report = auditor.audit(flipped, exp)
    assert not report.is_faithful
    assert any("polarity-flip" in c for c in report.unsupported_claims)
    assert report.metadata["audit_mode"] == "claim-binding"


def test_auditor_catches_wrong_part():
    """'a black belly' binds the right value to the wrong part; the bound
    claim never matches the certified conditions."""
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    conds = [
        condition("belly:belly_color=white", "<=", obs=0.1, idx=0),
        condition("beak:bill_color=black", ">", obs=0.9, idx=1),
    ]
    exp = explanation(conds)
    text = (
        "The model classifies this bird as Species X because it observed "
        "(1) no white belly color, (2) a black belly."
    )
    report = auditor.audit(text, exp)
    assert not report.is_faithful
    assert "beak:bill_color=black" in report.omitted_conditions


def test_auditor_accepts_pipeline_output_in_claim_mode():
    """The verbalizer's own output (negated attribute, has_part, relational)
    must audit faithful with the binding-based mode."""
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    conds = [
        condition("belly:belly_color=white", "<=", obs=0.1, idx=0),
        condition("has_part(crown)", ">", obs=1.0, idx=1),
        condition("above(crown, beak)", "<=", obs=0.0, idx=2),
        condition("beak:bill_color=black", ">", obs=0.9, idx=3),
    ]
    exp = explanation(conds)
    gen = ExplanationTemplateGenerator()
    text = gen.generate_sufficient_explanation(exp, matched(conds))
    report = auditor.audit(text, exp)
    assert report.is_faithful, report.audit_summary
    assert report.metadata["audit_mode"] == "claim-binding"


def test_auditor_foreign_text_falls_back_to_lexical_mode():
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    conds = [condition("belly:belly_color=white", "<=", obs=0.1)]
    report = auditor.audit("This bird clearly has a red tail.", explanation(conds))
    assert report.metadata["audit_mode"] == "lexical"
    assert not report.is_faithful


# --------------------------------------- round 2: absence-clause confidence ----
def test_resolver_reports_no_fake_confidence_without_provenance():
    """A fact without provenance (region unobserved) must yield
    card.confidence None — never the binary observed_value dressed up as
    reader confidence, which the complement logic would turn into a
    fabricated '[confidence: 1.00]'."""
    from grounded_vision.predicates.schema import Fact, FactDatabase, PredicateType
    from grounded_vision.explanations.evidence import EvidenceResolver

    db = FactDatabase(sample_id=1)
    db.add_fact(Fact(
        predicate="has_attribute", subject_id="part_belly_0",
        attribute="belly_color", value="white",
        confidence=0.1, truth_value=False,
        provenance=None, predicate_type=PredicateType.HAS_ATTRIBUTE,
    ))
    cond = condition("belly:belly_color=white", "<=", obs=0.0)
    card = EvidenceResolver(db).resolve_condition(cond)

    assert card.confidence is None
    assert card.bbox is None
    gen = ExplanationTemplateGenerator()
    clause = gen.verbalize_condition(card, cond)
    assert clause.startswith("no white belly color")
    assert "confidence" not in clause


def test_closing_sentence_states_predicted_concept_values():
    conds = [condition("beak:bill_color=black", ">", obs=0.9)]
    text = ExplanationTemplateGenerator().generate_sufficient_explanation(
        explanation(conds), matched(conds)
    )
    assert "given the predicted concept values" in text
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
