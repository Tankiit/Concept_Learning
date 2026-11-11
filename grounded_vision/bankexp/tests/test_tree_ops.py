"""Unit tests for utils/tree_ops.py (numpy only, no fixtures)."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.tree_ops import TreeOps  # noqa: E402


def make_tree():
    """(f0 AND f1) -> class 1 else 0, in the JSON export format."""
    return {
        "classes": [0, 1],
        "feature_names": ["f1", "f2"],
        "n_features": 2,
        "root": {"f": 0, "t": 0.5,
                 "l": {"c": 0},
                 "r": {"f": 1, "t": 0.5, "l": {"c": 0}, "r": {"c": 1}}},
    }


def test_predict():
    ops = TreeOps(make_tree())
    assert ops.predict_one(np.array([1, 1])) == 1
    assert ops.predict_one(np.array([0, 1])) == 0
    assert ops.predict_one(np.array([1, 0])) == 0
    assert list(ops.predict(np.array([[1, 1], [0, 1]]))) == [1, 0]


def test_path_literals():
    ops = TreeOps(make_tree())
    assert ops.path_literals(np.array([1, 1])) == [(0, 1), (1, 1)]
    assert ops.path_literals(np.array([0, 0])) == [(0, 0)]


def test_forces_sufficiency():
    ops = TreeOps(make_tree())
    assert ops.forces({0: 1, 1: 1}, 1)          # both fixed -> class 1
    assert not ops.forces({0: 1}, 1)            # f1 free can reach class 0
    assert ops.forces({0: 0}, 0)                # f0=0 forces class 0
    assert ops.sufficient([(0, 1), (1, 1)], 1)
    assert not ops.sufficient([(0, 1)], 1)


def test_minimal_reason():
    ops = TreeOps(make_tree())
    # for (1,1) the minimal sufficient reason is both literals
    assert ops.minimal_reason(np.array([1, 1]), 1) == [(0, 1), (1, 1)]
    # for (0,1) the single literal f0=0 suffices
    assert ops.minimal_reason(np.array([0, 1]), 0) == [(0, 0)]


def test_three_class_reachable_leaves():
    t = {"classes": [0, 1, 2], "feature_names": ["a", "b"], "n_features": 2,
         "root": {"f": 0, "t": 0.5,
                  "l": {"c": 2},
                  "r": {"f": 1, "t": 0.5, "l": {"c": 0}, "r": {"c": 1}}}}
    ops = TreeOps(t)
    # b=1 alone does NOT force class 1: a=0 still reaches the class-2 leaf
    assert not ops.forces({1: 1}, 1)
    assert ops.forces({0: 1, 1: 1}, 1)
    assert not ops.forces({}, 1)                # nothing fixed: all reachable
    assert ops.minimal_reason(np.array([1, 1]), 1) == [(0, 1), (1, 1)]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
