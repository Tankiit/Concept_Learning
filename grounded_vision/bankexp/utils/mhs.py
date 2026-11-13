"""
mhs.py — Minimal Hitting Set (MHS) utilities.

Port of CSU-TrustLab/behavior-explainer utils/mhs.py (stdlib + numpy;
no pysat dependency here).  A hitting set of a collection A is a set
that intersects every set in A; a minimal hitting set has no redundant
elements.

Solvers
-------
    minimal_hitting_setA(sets, _)   exact: smallest-cardinality hitting
                                    set (increasing-size search; falls
                                    back to greedy if the universe is
                                    too large for the cap)
    minimal_hitting_setB(sets)      random: seeded random greedy, then
                                    pruned to irreducible
    minimal_hitting_setC(sets, blocks, attempts, seed)
                                    hybrid: B with retries avoiding the
                                    blocked list of forbidden answers

Signatures mirror the upstream module so tests/test_mhs.py (ported
from upstream) run unmodified.
"""

from __future__ import annotations
import itertools
from collections.abc import Iterable

EXACT_CAP = 6  # max hitting-set cardinality for the exhaustive search


def _prune(hs: list, sets: list) -> list:
    """Remove redundant elements while the set still hits everything."""
    hs = list(hs)
    for e in list(hs):
        if all(any(x != e and x in hs for x in s) for s in sets):
            hs.remove(e)
    return hs


def minimal_hitting_setA(sets: list, _unused=None) -> list:
    """Exact smallest hitting set (bitmask increasing-size search up to
    EXACT_CAP); greedy-irreducible fallback past the cap."""
    sets = [set(s) for s in sets if s]
    if not sets:
        return []
    universe = sorted(set().union(*sets))
    elem_index = {e: i for i, e in enumerate(universe)}
    full = (1 << len(sets)) - 1
    masks = {}
    for e in universe:
        m = 0
        for si, s in enumerate(sets):
            if e in s:
                m |= 1 << si
        masks[e] = m
    for k in range(1, min(EXACT_CAP, len(universe)) + 1):
        for cand in itertools.combinations(universe, k):
            acc = 0
            for e in cand:
                acc |= masks[e]
                if acc == full:
                    break
            if acc == full:
                return list(cand)
    # fallback: greedy irreducible (not guaranteed minimum)
    hs = []
    remaining = [s for s in sets]
    while remaining:
        counts = {}
        for s in remaining:
            for e in s:
                counts[e] = counts.get(e, 0) + 1
        best = max(counts, key=lambda e: (counts[e], str(e)))
        hs.append(best)
        remaining = [s for s in remaining if best not in s]
    return _prune(hs, sets)


def minimal_hitting_setB(sets: list, seed: int = 0) -> list:
    """Random greedy hitting set, pruned to irreducible."""
    import random
    rng = random.Random(seed)
    if not sets:
        return []
    hs, remaining = [], [set(s) for s in sets if s]
    while remaining:
        e = rng.choice(sorted(set().union(*remaining)))
        hs.append(e)
        remaining = [s for s in remaining if e not in s]
    return _prune(hs, [set(s) for s in sets if s])


def minimal_hitting_setC(sets: list, blocks: Iterable,
                         attempts: int = 100, seed: int = 0) -> list:
    """Hybrid: random greedy with retries avoiding blocked answers."""
    blocks = [set(b) for b in blocks]
    sets = [set(s) for s in sets if s]
    for i in range(max(1, attempts)):
        hs = set(minimal_hitting_setB(sets, seed=seed + i))
        if hs not in blocks:
            return sorted(hs)
    return sorted(minimal_hitting_setB(sets, seed=seed + attempts))
