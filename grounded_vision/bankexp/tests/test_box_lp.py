"""Unit tests for utils/box_lp.py (hand-computable linear teachers)."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.box_lp import box_max_margin, reason_sufficient_exact  # noqa: E402


def two_class_teacher(w0, w1, b0=0.0, b1=0.0):
    return (np.array([w0, w1], dtype=np.float64),
            np.array([b0, b1], dtype=np.float64),
            np.array([0, 1]))


def test_dominant_literal_is_sufficient():
    # class0 logit 2*x0, class1 logit 0: fixing x0=1 wins everywhere
    W, b, c = two_class_teacher([2.0, 0.0], [0.0, 0.0])
    ok, worst = reason_sufficient_exact(W, b, c, [(0, 1)], 0)
    assert ok and worst == -2.0
    # fixing x0=0 leaves margin 0 (tie) -> not strictly sufficient
    ok, worst = reason_sufficient_exact(W, b, c, [(0, 0)], 0)
    assert not ok and worst == 0.0


def test_free_feature_breaks_sufficiency():
    # class0 logit x0, class1 logit x1: any single fixed feature leaves
    # the other free to tie or win -> never strictly sufficient
    W, b, c = two_class_teacher([1.0, 0.0], [0.0, 1.0])
    ok, _ = reason_sufficient_exact(W, b, c, [(0, 1)], 0)
    assert not ok
    ok, worst = reason_sufficient_exact(W, b, c, [(0, 1), (1, 0)], 0)
    assert ok and worst == -1.0


def test_bias_counted():
    # class1 has bias +3: class0 needs 4*x0 fixed at 1 to win the box
    W, b, c = two_class_teacher([4.0, 0.0], [0.0, 0.0], b1=3.0)
    ok, worst = reason_sufficient_exact(W, b, c, [(0, 1)], 0)
    assert ok and worst == -1.0
    ok, worst = reason_sufficient_exact(W, b, c, [(0, 0)], 0)
    assert not ok and worst == 3.0


def test_box_max_margin_vertex():
    # margin_d(x) = x1 - x0 over the box with x0 fixed 0: max at x1=1
    W, b, c = two_class_teacher([1.0, 0.0], [0.0, 1.0])
    m = box_max_margin(W, b, [(0, 0)], c_idx=0, d_idx=1)
    assert m == 1.0
    m = box_max_margin(W, b, [(0, 1)], c_idx=0, d_idx=1)
    assert m == 0.0  # x0=1 fixed: 1 - 1


def test_three_classes_worst_case():
    W = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    b = np.zeros(3)
    classes = np.array([0, 1, 2])
    # fixing x0=1: class1 margin = -x0+x1 -> 0 at x1=1 (tie); class2
    # margin = -0.5*x0+0.5*x1 -> 0 at x1=1.  Worst = 0 tie: insufficient.
    ok, worst = reason_sufficient_exact(W, b, classes, [(0, 1)], 0)
    assert not ok and worst == 0.0
    # fixing both: class1 margin -1, class2 margin -0.5 -> sufficient
    ok, worst = reason_sufficient_exact(W, b, classes,
                                        [(0, 1), (1, 0)], 0)
    assert ok and worst == -0.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
