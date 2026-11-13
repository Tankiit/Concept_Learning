"""
Unit Tests: three-valued predicate semantics (review requirement 1)
====================================================================
1. An attribute of an UNOBSERVED part produces NO fact (unknown), never a
   silently negative attribute.
2. An attribute of an observed part produces a positive fact (reader >=
   threshold) or a negative fact (reader < threshold).
3. The FeatureCompiler maps present -> +1, absent -> -1, unknown -> 0, so
   tree thresholds at -0.5/+0.5 separate the three states.
"""

import numpy as np

from grounded_vision.data.cub import CANONICAL_BOUND_ATTRIBUTES
from grounded_vision.perception.regions import extract_part_regions
from grounded_vision.perception.readers import PerceptionOutput
from grounded_vision.predicates.grounding import ground_perception_output
from grounded_vision.predicates.schema import PredicateType
from grounded_vision.reasoning.features import (
    FeatureCompiler, RepresentationMode)


def _perception(parts_visible, attr_probs):
    """regions only for visible parts; attr_probs: {cub_attr_id: prob}."""
    boxes = {p: (0.1, 0.1, 0.4, 0.4) for p in parts_visible}
    vis = {p: (1.0 if p in parts_visible else 0.0) for p in
           ["beak", "breast", "belly", "crown"]}
    regions = extract_part_regions(boxes, vis)
    return PerceptionOutput(
        sample_id=0, regions=regions, part_visibilities=vis,
        attribute_probabilities=dict(attr_probs), prototype_matches={},
        feature_embeddings={}, metadata={})


def _attrs_of(part):
    """[(cub_attr_id, canonical_tuple)] for the first 2 DISTINCT attribute
    names of part (canonical tuples repeat names across values)."""
    seen, out = set(), []
    for a in CANONICAL_BOUND_ATTRIBUTES:
        if a[1] == part and a[2] not in seen:
            seen.add(a[2])
            out.append((a[0], a))
        if len(out) == 2:
            break
    return out


def test_unobserved_part_emits_no_attribute_facts():
    # beak observed, belly NOT observed: belly attributes must be absent
    # from the database entirely (unknown), while beak attributes exist.
    probs = {a[0]: 0.9 for a in CANONICAL_BOUND_ATTRIBUTES if a[1] == "beak"}
    perc = _perception(["beak"], probs)
    db = ground_perception_output(perc, probability_threshold=0.5,
                                  include_relations=False)
    attr_facts = [f for f in db.facts.values()
                  if f.predicate_type == PredicateType.HAS_ATTRIBUTE]
    subjects = {f.subject_id for f in attr_facts}
    assert "part_beak_0" in subjects
    assert "part_belly_0" not in subjects, \
        "unobserved part must not produce attribute facts (unknown != absent)"
    assert all(f.truth_value for f in attr_facts if f.subject_id == "part_beak_0")


def test_observed_part_emits_positive_and_negative_facts():
    (hi_id, hi_t), (lo_id, lo_t) = _attrs_of("beak")
    perc = _perception(["beak"], {hi_id: 0.9, lo_id: 0.1})
    db = ground_perception_output(perc, probability_threshold=0.5,
                                  include_relations=False)
    attr_facts = {f"{f.attribute}={f.value}": f for f in db.facts.values()
                  if f.predicate_type == PredicateType.HAS_ATTRIBUTE}
    assert attr_facts[f"{hi_t[2]}={hi_t[3]}"].truth_value is True
    assert attr_facts[f"{lo_t[2]}={lo_t[3]}"].truth_value is False


def test_compiler_maps_present_absent_unknown():
    (hi_id, hi_t), (lo_id, lo_t) = _attrs_of("beak")
    # beak observed with one present + one absent attribute; belly unobserved
    perc = _perception(["beak"], {hi_id: 0.9, lo_id: 0.1})
    db = ground_perception_output(perc, probability_threshold=0.5,
                                  include_relations=False)
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND,
                               attributes=list(CANONICAL_BOUND_ATTRIBUTES))
    vec = compiler.compile_database(db)
    hi = compiler.feature_to_idx[f"beak:{hi_t[2]}={hi_t[3]}"]
    lo = compiler.feature_to_idx[f"beak:{lo_t[2]}={lo_t[3]}"]
    belly_feat = next(i for i, n in enumerate(compiler.feature_names)
                      if n.startswith("belly:"))
    assert vec[hi] == 1.0   # present
    assert vec[lo] == -1.0  # observed and absent
    assert vec[belly_feat] == 0.0  # unobserved -> unknown, not negative


def test_tristate_separates_in_a_tree():
    # a split at threshold 0 isolates 'present' from {absent, unknown};
    # a split at -0.5 isolates 'absent' from {unknown, present}.
    from grounded_vision.reasoning.tree_head import CertifiableTreeHead
    rng = np.random.RandomState(0)
    X = rng.choice([-1.0, 0.0, 1.0], size=(300, 4))
    # class 1 iff feature 0 == 1 (present)
    y = (X[:, 0] == 1.0).astype(int)
    head = CertifiableTreeHead(max_depth=3, feature_names=[f"f{i}" for i in range(4)])
    head.fit(X, y)
    conds = [c for c in head.get_decision_path(np.array([1.0, 0, 0, 0]))
             if c.feature_idx == 0]
    assert conds and all(c.threshold in (0.0, -0.5, 0.5) or abs(c.threshold) <= 0.5
                         for c in conds)
