"""Exact box-LP sufficiency for a linear head (numpy only).

A reason R (literals fixing bank features to 0/1) is EXACTLY sufficient
for class c under a linear teacher iff, for every competing class d,
the maximum of margin_d(x) = (w_d - w_c) . x + (b_d - b_c) over the
completion box ([0,1] with R's coordinates fixed) is negative.  The
objective is linear over a box, so the maximum sits at a vertex: each
free coordinate independently takes 1 if its coefficient is positive,
else 0.  Exact and polynomial -- no solver, no sampling.

Shared by analysis/margin_sufficiency.py (metric) and src/conaxp.py
(erasure oracle).
"""

from __future__ import annotations
import numpy as np


def box_max_margin(W, b, literals, c_idx, d_idx):
    """max over box completions of margin_d for class pair (c, d)."""
    diff = W[d_idx] - W[c_idx]
    base = float(b[d_idx] - b[c_idx])
    x = np.zeros(len(diff))
    free = np.ones(len(diff), dtype=bool)
    for f, bit in literals:
        x[f] = bit
        free[f] = False
    x[free & (diff > 0)] = 1.0
    x[free & (diff <= 0)] = 0.0
    return base + float(diff @ x)


def reason_sufficient_exact(W, b, classes, literals, target, eps=0.0):
    """Exact box sufficiency for the linear teacher.  Returns
    (bool, worst_margin); worst_margin is the max over competing classes
    of the box-max margin (negative == safe)."""
    c_idx = int(np.where(classes == target)[0][0])
    worst = -np.inf
    for d in range(len(classes)):
        if d == c_idx:
            continue
        m = box_max_margin(W, b, literals, c_idx, d)
        worst = max(worst, m)
    return bool(worst < -eps), float(worst)
