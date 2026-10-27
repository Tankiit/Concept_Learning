"""
Unit Tests: Formal Sufficiency Verification and Explanation Auditing
===================================================================
Verifies:
1. Mathematical sufficiency checking via TreeSufficiencyVerifier
2. Prime implicant condition reduction
3. PyXAI backend integration
4. Explanation faithfulness auditing (omitted vs. unsupported claims)
"""

import pytest
import numpy as np

from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.schema import (
    ExplanationCondition,
    SufficientExplanation,
    VerificationStatus,
)
from grounded_vision.explanations.tree_verifier import (
    TreeSufficiencyVerifier,
    extract_minimal_sufficient_conditions,
)
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.language.checks import ExplanationAuditor


@pytest.fixture
def trained_tree():
    # 3 features:
    # f0: beak:bill_shape=cone
    # f1: crown:crown_color=black
    # f2: belly:belly_color=yellow
    feat_names = [
        "beak:bill_shape=cone",
        "crown:crown_color=black",
        "belly:belly_color=yellow",
    ]
    # Class 0: f0 > 0.5 and f2 > 0.5
    # Class 1: f0 <= 0.5
    # Class 2: f0 > 0.5 and f2 <= 0.5
    X = np.array([
        [1.0, 0.0, 1.0],  # 0
        [1.0, 1.0, 1.0],  # 0
        [0.0, 1.0, 0.0],  # 1
        [0.0, 0.0, 1.0],  # 1
        [1.0, 0.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 2
    ], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2], dtype=int)

    tree = CertifiableTreeHead(
        max_depth=4,
        min_samples_leaf=1,
        feature_names=feat_names,
        class_names=["Species_Zero", "Species_One", "Species_Two"],
    )
    tree.fit(X, y)
    return tree


def test_tree_sufficiency_verifier_exact(trained_tree):
    verifier = TreeSufficiencyVerifier(trained_tree.clf, trained_tree.feature_names)

    # Path to Class 0 requires f0 > 0.5 and f2 > 0.5
    cond_f0_gt = ExplanationCondition(0, "beak:bill_shape=cone", ">", 0.5, 1.0)
    cond_f2_gt = ExplanationCondition(2, "belly:belly_color=yellow", ">", 0.5, 1.0)

    # 1. Full sufficient set -> Must be VERIFIED
    cert_full = verifier.verify_sufficiency([cond_f0_gt, cond_f2_gt], target_class=0)
    assert cert_full.status == VerificationStatus.VERIFIED

    # 2. Incomplete condition set (only f0 > 0.5) -> Leaves open Species 2 -> Must be FALSIFIED
    cert_incomplete = verifier.verify_sufficiency([cond_f0_gt], target_class=0)
    assert cert_incomplete.status == VerificationStatus.FALSIFIED
    assert len(cert_incomplete.proof_details["conflicting_leaves"]) > 0


def test_minimal_sufficient_extraction(trained_tree):
    # Instance with [1.0, 1.0, 1.0] -> Class 0. Note f1 is irrelevant to tree decision.
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    suff_exp = extract_minimal_sufficient_conditions(trained_tree, x, target_class=0)

    assert suff_exp.certificate.status == VerificationStatus.VERIFIED
    # Should only contain the 2 relevant features (f0 and f2), pruning any superfluous tests
    assert suff_exp.sparsity == 2
    f_indices = {c.feature_idx for c in suff_exp.conditions}
    assert f_indices == {0, 2}


def test_pyxai_backend_and_contrastive(trained_tree):
    backend = PyXAIBackend(trained_tree)
    x = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    suff_exp = backend.explain_sufficient(x, target_class=0)
    assert suff_exp.certificate.status == VerificationStatus.VERIFIED

    contrast_exp = backend.explain_contrastive(x, original_class=0, contrast_class=2)
    assert contrast_exp is not None
    assert contrast_exp.contrast_class == 2
    # To become class 2, f2 (belly yellow) needs to be flipped to <= 0.5
    altered_feats = [c.feature_idx for c in contrast_exp.conditions_to_alter]
    assert 2 in altered_feats


def test_explanation_auditor_faithfulness():
    auditor = ExplanationAuditor()
    cond1 = ExplanationCondition(0, "beak:bill_shape=cone", ">", 0.5, 1.0)
    cond2 = ExplanationCondition(2, "belly:belly_color=yellow", ">", 0.5, 1.0)

    suff_exp = SufficientExplanation(
        sample_id=0,
        target_class=0,
        target_class_name="Species_Zero",
        conditions=[cond1, cond2],
        certificate=None,
    )

    # 1. Perfectly faithful text mentioning both verified conditions and no hallucinations
    faithful_text = "The bird has a cone bill and a yellow belly."
    rep1 = auditor.audit(faithful_text, suff_exp)
    assert rep1.is_faithful
    assert len(rep1.omitted_conditions) == 0
    assert len(rep1.unsupported_claims) == 0

    # 2. Text omitting the belly condition
    omitted_text = "The bird has a cone bill."
    rep2 = auditor.audit(omitted_text, suff_exp)
    assert not rep2.is_faithful
    assert len(rep2.omitted_conditions) == 1
    assert "belly" in rep2.omitted_conditions[0]

    # 3. Text hallucinating red wings
    hallucinated_text = "The bird has a cone bill, a yellow belly, and red wings."
    rep3 = auditor.audit(hallucinated_text, suff_exp)
    assert not rep3.is_faithful
    assert any("red" in claim for claim in rep3.unsupported_claims)
