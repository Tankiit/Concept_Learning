"""numpy-only operations on exported binary decision trees.

Shared by src/explain.py (independent verifier) and analysis/robustness.py.
Tree JSON contract (written by src/distill.py):

    {"classes": [...], "feature_names": [...], "n_features": int,
     "root": {"f": <int>, "t": <float>, "l": <node>, "r": <node>}
     or leaf {"c": <class label>}}

Binary-input convention (matches sklearn on 0/1 features and the PyXAI
EQ-conversion): feature value <= t goes left (bit 0), value > t goes
right (bit 1).
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def flatten(root: dict):
    """nested JSON tree -> arrays (feature, left, right, leaf class)."""
    feats, lefts, rights, preds = [], [], [], []

    def rec(node):
        nid = len(feats)
        feats.append(-1); lefts.append(-1); rights.append(-1); preds.append(None)
        if "c" in node and "f" not in node:
            preds[nid] = node["c"]
            return nid
        feats[nid] = int(node["f"])
        lefts[nid] = rec(node["l"])
        rights[nid] = rec(node["r"])
        return nid

    rec(root)
    return (np.array(feats, dtype=np.int64),
            np.array(lefts, dtype=np.int64),
            np.array(rights, dtype=np.int64),
            preds)


class TreeOps:
    """Exact routing / sufficiency for a binary-input tree (numpy only)."""

    def __init__(self, tree_json: dict):
        self.classes = list(tree_json["classes"])
        self.feature_names = list(tree_json["feature_names"])
        self.n_features = int(tree_json["n_features"])
        f, l, r, p = flatten(tree_json["root"])
        self.f, self.l, self.r, self.pred = f, l, r, p

    def predict_one(self, x) -> int:
        nid = 0
        while self.pred[nid] is None:
            nid = int(self.r[nid]) if x[int(self.f[nid])] > 0.5 else int(self.l[nid])
        return self.pred[nid]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(X[i]) for i in range(len(X))])

    def path_literals(self, x):
        """[(feature_idx, bit)] along the decision path."""
        nid, out = 0, []
        while self.pred[nid] is None:
            f = int(self.f[nid])
            bit = 1 if x[f] > 0.5 else 0
            out.append((f, bit))
            nid = int(self.r[nid] if bit else self.l[nid])
        return out

    def forces(self, assign: dict, target) -> bool:
        """True iff every leaf reachable under the partial assignment
        {feature -> 0/1} predicts target and at least one is reachable."""
        stack = [0]
        reached = False
        while stack:
            nid = stack.pop()
            if self.pred[nid] is not None:
                if self.pred[nid] != target:
                    return False
                reached = True
            else:
                f = int(self.f[nid])
                if f in assign:
                    stack.append(int(self.r[nid] if assign[f] else self.l[nid]))
                else:
                    stack.append(int(self.l[nid]))
                    stack.append(int(self.r[nid]))
        return reached

    def sufficient(self, literals, target) -> bool:
        return self.forces({f: b for f, b in literals}, target)

    def minimal_reason(self, x, target):
        """Backward-elimination minimal sufficient subset of the decision
        path literals (the committed native extraction, numpy-only)."""
        lits = self.path_literals(x)
        keep = list(lits)
        for lit in lits:
            trial = [l for l in keep if l != lit]
            if self.forces({f: b for f, b in trial}, target):
                keep = trial
        return keep


def load_tree(path) -> TreeOps:
    return TreeOps(json.loads(Path(path).read_text()))
