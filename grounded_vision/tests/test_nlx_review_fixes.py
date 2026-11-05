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
    # no recognizable value words -> no heuristic claims -> lexical fallback
    report = auditor.audit("This bird is large and perched.", explanation(conds))
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


# --------------------------------------------- round 3: free-text claims -------
def test_heuristic_claims_extracted_from_free_text():
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    claims = auditor.extract_text_claims(
        "The bird has a black bill, a yellow breast, and no white belly."
    )
    got = {(c["part"], c["value"], c["polarity"]) for c in claims}
    # surface part words ('bill'); beak/bill aliasing happens at match time
    assert ("bill", "black", "present") in got
    assert ("breast", "yellow", "present") in got
    assert ("belly", "white", "absent") in got


def test_heuristic_audit_catches_flipped_foreign_text():
    """LLM arm: 'a white belly' against a certified absence must fail in
    claim-heuristic mode, not just lexically."""
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    conds = [condition("belly:belly_color=white", "<=", obs=0.1)]
    exp = explanation(conds)
    report = auditor.audit(
        "This individual shows a white belly and a stout bill.", exp
    )
    assert report.metadata["audit_mode"] == "claim-heuristic"
    assert not report.is_faithful
    assert any(c.startswith("claim:") for c in report.unsupported_claims)


def test_heuristic_audit_accepts_faithful_foreign_text():
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    conds = [
        condition("beak:bill_color=black", ">", obs=0.9, idx=0),
        condition("belly:belly_color=white", "<=", obs=0.1, idx=1),
    ]
    report = auditor.audit(
        "The bird has a black bill and no white belly.", explanation(conds)
    )
    assert not report.unsupported_claims, report.unsupported_claims


def test_audit_mode_split_in_explanation_suite():
    from grounded_vision.evaluation.explanations import evaluate_explanation_suite

    gen = ExplanationTemplateGenerator()
    conds = [condition("belly:belly_color=white", "<=", obs=0.1)]
    exp = explanation(conds)
    good = gen.generate_sufficient_explanation(exp, matched(conds))

    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    reports = [
        auditor.audit(good, exp),                              # claim-binding
        auditor.audit("This bird has a red tail.", exp),       # claim-heuristic
    ]
    metrics = evaluate_explanation_suite([exp, exp], reports)
    modes = metrics["nl_audit_modes"]
    assert set(modes) == {"claim-binding", "claim-heuristic"}
    assert modes["claim-binding"] == {"n": 1, "faithful_rate": 1.0, "hallucination_rate": 0.0}
    assert modes["claim-heuristic"]["n"] == 1
    assert modes["claim-heuristic"]["faithful_rate"] == 0.0


# ------------------------------------------------- round 3: parts.txt fix -----
def test_cub_part_names_use_underscores(tmp_path):
    """CUB's parts.txt writes 'left wing' with a space; the parser must
    normalize to left_wing or the part never matches the vocabulary and
    reads as unobserved (6 of 15 parts were invisible before this fix).
    Hermetic: builds a minimal CUB annotation fixture."""
    from PIL import Image
    from grounded_vision.data.cub import _parse_cub_annotations_cached, CUB_PARTS

    root = tmp_path / "CUB_200_2011"
    (root / "images" / "03.Warbler").mkdir(parents=True)
    Image.new("RGB", (60, 40)).save(root / "images" / "03.Warbler" / "a.jpg")

    (root / "images.txt").write_text("1 03.Warbler/a.jpg\n")
    (root / "classes.txt").write_text("3 Warbler\n")
    (root / "image_class_labels.txt").write_text("1 3\n")
    (root / "train_test_split.txt").write_text("1 0\n")
    (root / "attributes.txt").write_text("1 has_bill_shape_dagger\n")
    (root / "attributes").mkdir()
    (root / "attributes" / "image_attribute_labels.txt").write_text("1 1 0\n")
    (root / "parts").mkdir()
    # real CUB names: spaces inside part names
    (root / "parts" / "parts.txt").write_text(
        "1 back\n2 beak\n3 belly\n4 breast\n5 crown\n6 forehead\n7 left eye\n"
        "8 left leg\n9 left wing\n10 nape\n11 right eye\n12 right leg\n"
        "13 right wing\n14 tail\n15 throat\n"
    )
    (root / "parts" / "part_locs.txt").write_text(
        "1 9 20.0 20.0 1\n1 2 30.0 20.0 1\n1 4 30.0 25.0 0\n"
    )

    class_names, records = _parse_cub_annotations_cached(str(root))
    parts = records[1]["parts"]
    assert set(parts) == {"left_wing", "beak", "breast"}
    for name in parts:
        assert name in CUB_PARTS
    assert parts["left_wing"][2] == 1     # visible left wing survives the parse
    assert parts["breast"][2] == 0        # invisible part recorded as 0


def test_auditor_ignores_color_words_in_species_name():
    """CUB species names carry color words ('Brown_Booby'); the class name is
    not a claim about the image and must not fail the lexical scan."""
    from grounded_vision.language.checks import ExplanationAuditor

    auditor = ExplanationAuditor()
    conds = [condition("belly:belly_color=white", "<=", obs=0.1)]
    exp = explanation(conds, target="Brown_Booby")
    gen = ExplanationTemplateGenerator()
    text = gen.generate_sufficient_explanation(exp, matched(conds))
    assert "Brown_Booby" in text
    report = auditor.audit(text, exp)
    assert report.is_faithful, report.audit_summary
    assert report.metadata["audit_mode"] == "claim-binding"
