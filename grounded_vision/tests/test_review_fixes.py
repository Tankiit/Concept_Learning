"""
Unit Tests: Review-driven validation fixes
==========================================
Regression tests for the correctness fixes from the external review:

1. Verifier class-label mapping — a leaf's argmax over class counts is a
   POSITION in ``clf.classes_``, not a label. Sufficiency must compare
   caller-supplied labels to labels.
2. Negated relational templates — a certified ABSENCE of a spatial relation
   must verbalize as absence.
3. Activation tile pooling locality — 4x4 tile pooling must average within
   tile blocks; averaging over the wrong axes turns tiles into row strips.
"""

import numpy as np

from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.tree_verifier import (
    extract_minimal_sufficient_conditions,
)
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.explanations.evidence import VisualEvidenceCard
from grounded_vision.explanations.schema import VerificationStatus
from grounded_vision.language.templates import ExplanationTemplateGenerator


def test_verifier_handles_noncontiguous_labels():
    """Labels {3, 7} are not contiguous: leaf positions are 0/1 but
    predictions are the labels themselves. Verification must succeed."""
    X = np.array(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    y = np.array([3, 3, 7, 7])
    tree = CertifiableTreeHead(
        max_depth=3, feature_names=["a:x=1", "b:y=1"]
    )
    tree.fit(X, y)
    backend = PyXAIBackend(tree)

    for x, expected in zip(X, [3, 3, 7, 7]):
        pred = int(tree.predict(x.reshape(1, -1))[0])
        assert pred == expected
        exp = extract_minimal_sufficient_conditions(tree, x, pred, sample_id=0)
        assert exp.certificate.status == VerificationStatus.VERIFIED
        backend_exp = backend.explain_sufficient(x, pred, sample_id=0)
        assert backend_exp.certificate.status == VerificationStatus.VERIFIED


def test_negated_relational_template_verbalizes_absence():
    """A condition `above(crown, beak) <= 0.5` certifies the crown is NOT
    above the beak; the template must say so."""
    gen = ExplanationTemplateGenerator()

    def card(condition_str):
        return VisualEvidenceCard(
            condition_str=condition_str,
            feature_name="above(crown, beak)",
            part_name="crown",
            bbox=None,
        )

    pos_text = gen.verbalize_condition(card("above(crown, beak) > 0.500"))
    neg_text = gen.verbalize_condition(card("above(crown, beak) <= 0.500"))

    assert "situated above" in pos_text
    assert "not situated" not in pos_text
    assert "not situated above" in neg_text


def test_activation_tile_pooling_is_local():
    """A patch delta at (row=2, col=13) must land in tile (r=0, c=3).
    Regression for run_activation_tree_grounding.py, where mean(axis=(4,5))
    averaged over ALL columns and produced full-width row strips."""
    fmap = np.zeros((1, 1, 16, 16), dtype=np.float32)
    fmap[0, 0, 2, 13] = 9.0

    pooled = (
        fmap.reshape(1, 1, 4, 4, 4, 4)
        .mean(axis=(3, 5))  # (row-in-block, col-in-block)
        .transpose(0, 2, 3, 1)
        .reshape(1, -1)
    )
    # feature index = (r * 4 + c) * channels + ch, with 1 channel here
    assert pooled[0, 0 * 4 + 3] == 9.0 / 16.0
    assert int((pooled != 0).sum()) == 1

    buggy = (
        fmap.reshape(1, 1, 4, 4, 4, 4)
        .mean(axis=(4, 5))
        .transpose(0, 2, 3, 1)
        .reshape(1, -1)
    )
    assert buggy[0, 0 * 4 + 3] == 0.0  # energy smeared outside the true tile
