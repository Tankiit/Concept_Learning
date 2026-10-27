"""
Unit Tests: Feature Compilation and Tree Classifier Export
==========================================================
Verifies:
1. Feature compilation across Flat, Bound, and Relational modes
2. Invertible dictionary lookup (index <-> feature string)
3. CertifiableTreeHead fitting and prediction
4. Decision path extraction and rule export
"""

import pytest
import numpy as np

from grounded_vision.predicates.schema import Fact, FactDatabase, PredicateType
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead, DecisionPathStep


def test_feature_compiler_modes():
    # 1. Flat mode
    comp_flat = FeatureCompiler(mode=RepresentationMode.FLAT)
    assert comp_flat.num_features > 0
    assert any("bill_shape=curved" in name for name in comp_flat.feature_names)
    assert not any("part_of" in name for name in comp_flat.feature_names)

    # 2. Bound mode
    comp_bound = FeatureCompiler(mode=RepresentationMode.BOUND)
    assert comp_bound.num_features > comp_flat.num_features
    assert any("beak:bill_shape=curved" in name for name in comp_bound.feature_names)
    assert any("has_part(beak)" in name for name in comp_bound.feature_names)

    # 3. Relational mode
    comp_rel = FeatureCompiler(mode=RepresentationMode.RELATIONAL)
    assert comp_rel.num_features > comp_bound.num_features
    assert any("above(crown, beak)" in name for name in comp_rel.feature_names)


def test_compilation_from_database():
    comp = FeatureCompiler(mode=RepresentationMode.BOUND)
    db = FactDatabase(sample_id=1)

    db.add_fact(
        Fact(
            predicate="has_attribute",
            subject_id="part_beak_0",
            attribute="bill_shape",
            value="curved",
            confidence=0.9,
            truth_value=True,
            predicate_type=PredicateType.HAS_ATTRIBUTE,
        )
    )
    db.add_fact(
        Fact(
            predicate="has_part",
            subject_id="bird_0",
            value="beak",
            confidence=1.0,
            truth_value=True,
            predicate_type=PredicateType.HAS_PART,
        )
    )

    vec_binary = comp.compile_database(db, use_probabilities=False)
    vec_prob = comp.compile_database(db, use_probabilities=True)

    idx_attr = comp.feature_to_idx["beak:bill_shape=curved"]
    idx_part = comp.feature_to_idx["has_part(beak)"]

    assert vec_binary[idx_attr] == 1.0
    assert vec_prob[idx_attr] == pytest.approx(0.9, abs=1e-4)
    assert vec_binary[idx_part] == 1.0


def test_tree_head_training_and_decision_path():
    comp = FeatureCompiler(mode=RepresentationMode.BOUND)
    n_feats = comp.num_features

    # Create synthetic dataset: Class 0 has feat 0, Class 1 has feat 1
    X = np.zeros((20, n_feats), dtype=np.float32)
    y = np.zeros(20, dtype=int)
    for i in range(10):
        X[i, 0] = 1.0
        y[i] = 0
    for i in range(10, 20):
        X[i, 1] = 1.0
        y[i] = 1

    tree = CertifiableTreeHead(
        max_depth=3,
        feature_names=comp.feature_names,
        class_names=["Species_A", "Species_B"],
    )
    tree.fit(X, y)

    # Test predictions
    preds = tree.predict(X)
    assert np.array_equal(preds, y)

    # Test decision path
    sample = X[0]
    steps = tree.get_decision_path(sample)
    assert len(steps) > 0
    assert isinstance(steps[0], DecisionPathStep)
    assert steps[0].feature_name in comp.feature_names

    # Test rule export
    rules = tree.export_rules()
    assert len(rules) >= 2
    for r in rules:
        assert "leaf_id" in r
        assert "predicted_class" in r
        assert "conditions" in r
