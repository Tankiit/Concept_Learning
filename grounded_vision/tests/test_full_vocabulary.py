"""
Tests for the configurable bound-attribute vocabulary (canonical 28 vs full
CUB-312 with whole-bird binding).
"""

import os
import pytest
import numpy as np
import torch

from grounded_vision.data.attributes import (
    BoundAttribute, get_bound_attributes, vocabulary_cub_ids, valid_parts,
)
from grounded_vision.data.cub import CUB_PARTS

CUB_DIR = os.environ.get(
    "GROUNDED_VISION_CUB_DIR",
    "/Users/cril/tanmoy/research/data/CUB_200_2011/CUB_200_2011",
)
HAS_REAL_DATA = os.path.exists(os.path.join(CUB_DIR, "attributes.txt"))


def test_mock_canonical_vocabulary_without_data_dir():
    """Canonical selection must work with no raw CUB files (hermetic path)."""
    vocab = get_bound_attributes("canonical", "/nonexistent/path")
    assert len(vocab) == 28
    assert vocab[0] == BoundAttribute(0, "beak", "bill_shape", "curved")


def test_unknown_attribute_set_rejected():
    with pytest.raises(ValueError):
        get_bound_attributes("mega", CUB_DIR)


def test_full_vocabulary_binding():
    if not HAS_REAL_DATA:
        pytest.skip("real CUB attributes.txt not available")
    vocab = get_bound_attributes("full", CUB_DIR)
    ids = vocabulary_cub_ids("full", CUB_DIR)
    # every raw CUB attribute covered, local ids dense
    assert len(set(ids)) == 312
    assert [a.attr_id for a in vocab] == list(range(len(vocab)))
    # all parts valid, whole-bird used
    assert {a.part for a in vocab} <= valid_parts(vocab)
    assert any(a.part == "bird" for a in vocab)
    # wing attributes bind to both wings (canonical convention)
    wing_attrs = [a for a in vocab if a.attr_name == "wing_color" and a.value == "black"]
    assert {a.part for a in wing_attrs} == {"left_wing", "right_wing"}


def test_compiler_bird_namespace_and_canonical_regression():
    from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode

    # Canonical default unchanged: 28 attributes + 8 part bits = 36 features
    canon = FeatureCompiler(mode=RepresentationMode.BOUND)
    assert canon.num_features == 36
    assert "beak:bill_shape=curved" in canon.feature_to_idx

    mini = [
        BoundAttribute(0, "beak", "bill_shape", "curved"),
        BoundAttribute(1, "bird", "size", "small"),
        BoundAttribute(2, "bird", "shape", "perching-like"),
    ]
    comp = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=mini)
    assert "bird:size=small" in comp.feature_to_idx
    assert "bird:shape=perching-like" in comp.feature_to_idx
    assert "has_part(beak)" in comp.feature_to_idx
    assert not any(n.startswith("has_part(bird") for n in comp.feature_names)


def test_grounding_emits_bird_subject_facts():
    from grounded_vision.predicates.grounding import ground_perception_output
    from grounded_vision.perception.readers import PerceptionOutput
    from grounded_vision.perception.regions import Region

    vocab = [
        BoundAttribute(0, "beak", "bill_shape", "curved"),
        BoundAttribute(1, "bird", "size", "small"),
    ]
    regions = [
        Region(id="region_beak", part_name="beak", bbox=(0.1, 0.4, 0.3, 0.8), confidence=1.0),
        Region(id="region_bird", part_name="bird", bbox=(0.0, 0.0, 1.0, 1.0), confidence=1.0),
    ]
    p_out = PerceptionOutput(
        sample_id=1, regions=regions,
        part_visibilities={"beak": 1.0, "bird": 1.0},
        attribute_probabilities={0: 0.9, 1: 0.8},
        prototype_matches={},
        feature_embeddings={},
    )
    db = ground_perception_output(p_out, probability_threshold=0.5, attributes=vocab)
    fact_keys = {f.key: f for f in db.facts.values()}
    size_key = next(k for k in fact_keys if "size" in k and "small" in k)
    assert fact_keys[size_key].subject_id == "bird_0"
    assert fact_keys[size_key].provenance.region_id == "region_bird"
    curved_key = next(k for k in fact_keys if "curved" in k)
    assert fact_keys[curved_key].subject_id.startswith("part_beak")


def test_reader_and_bank_respect_vocabulary():
    from grounded_vision.perception.readers import AttributeReader
    from grounded_vision.perception.prototypes import AttributePrototypeBank

    mini = [
        BoundAttribute(0, "beak", "bill_shape", "curved"),
        BoundAttribute(1, "bird", "size", "small"),
    ]
    reader = AttributeReader(embedding_dim=16, attributes=mini)
    assert set(reader.classifiers.keys()) == {"attr_0", "attr_1"}
    bank = AttributePrototypeBank(embedding_dim=16, num_prototypes_per_attr=2, attributes=mini)
    assert set(bank.prototypes.keys()) == {0, 1}
