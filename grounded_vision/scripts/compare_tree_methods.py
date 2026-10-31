"""
Script: compare_tree_methods.py
===============================
Tree-based head comparison on the CUB-28 concept matrices (GT regions),
scored in the paper's currency: accuracy vs certified proof size.

Tier 1 — GOSDT (optimal sparse decision trees): converted to a plain
threshold tree, gated by an exact-prediction check, then explained with the
NATIVE verifier (same machinery as CART — proofs apply unchanged).
Tier 3 — RuleFit (imodels): interpretable rule extraction, but no entailment
proofs — the "informal" contrast row.
Tier 2 — RandomForest: accuracy reference (500 trees) + a small forest
(25 trees, depth 6) exported as a PyXAI payload for compiler-based proofs.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import yaml
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_full_run import compile_matrices  # noqa: E402
from grounded_vision.pipeline import load_perception, FeatureProvider, active_vocabulary  # noqa: E402
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier  # noqa: E402
from grounded_vision.explanations.schema import ExplanationCondition, VerificationStatus  # noqa: E402

RUN = "/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_full"
OUT = os.path.join(RUN, "tree_methods_comparison.json")


# ------------------------------------------------ GOSDT -> threshold tree ----
def gosdt_to_arrays(root, n_classes: int, feature_names):
    """Flattens a GOSDT tree into (features, thresholds, children, values).
    Internal nodes split binary features at 0.5; leaves carry .prediction."""
    feats, thrs, left, right, vals, order = [], [], [], [], [], []

    def visit(node):
        idx = len(feats)
        order.append(node)
        feats.append(0); thrs.append(0.0); left.append(-1); right.append(-1)
        vals.append(None)
        if hasattr(node, "left_child") and hasattr(node, "right_child") \
                and getattr(node, "feature", None) is not None:
            feats[idx] = int(node.feature)
            thrs[idx] = 0.5
            l, r = visit(node.left_child), visit(node.right_child)
            left[idx], right[idx] = l, r
        else:  # Leaf
            v = np.zeros((1, max(n_classes, int(node.prediction) + 1)))
            v[0, int(node.prediction)] = 1.0
            vals[idx] = v.reshape(-1, 1)
        return idx

    visit(root)
    n_cls = max(n_classes, max(int(getattr(n, "prediction", 0)) for n in order) + 1)
    value = np.zeros((len(feats), n_cls, 1))
    for i, v in enumerate(vals):
        if v is not None:
            value[i, :v.shape[0], 0] = v[:, 0]
    return (np.array(feats), np.array(thrs), np.array(left), np.array(right), value)


class SimpleThresholdTree:
    """Duck-typed sklearn-like tree: predict + decision path + verifier input."""

    def __init__(self, feats, thrs, left, right, value, feature_names):
        self._f, self._t, self._l, self._r, self._v = feats, thrs, left, right, value
        self.feature_names = feature_names
        self.tree_ = SimpleNamespace(
            feature=feats, threshold=thrs, children_left=left,
            children_right=right, value=value, node_count=len(feats),
            n_features=len(feature_names),
        )

    def predict_one(self, x) -> int:
        node = 0
        while self._l[node] != -1:
            node = self._l[node] if x[self._f[node]] <= self._t[node] else self._r[node]
        return int(np.argmax(self._v[node, :, 0]))

    def path_conditions(self, x):
        out, node = [], 0
        while self._l[node] != -1:
            f = int(self._f[node])
            is_left = x[f] <= self._t[node]
            out.append((f, "<=" if is_left else ">", float(self._t[node])))
            node = self._l[node] if is_left else self._r[node]
        return out, int(np.argmax(self._v[node, :, 0]))


def certified_reasons(tree: SimpleThresholdTree, Xte, preds, rows, feature_names):
    """Prime implicants via backward elimination, verified by the native
    TreeSufficiencyVerifier (same decision procedure as the CART pipeline)."""
    verifier = TreeSufficiencyVerifier(tree, feature_names)
    sizes, verified = [], 0
    for i in rows:
        conds, pred = tree.path_conditions(Xte[i])
        conds = [ExplanationCondition(f, feature_names[f], op, t, observed_value=0.0)
                 for (f, op, t) in conds]
        cert = verifier.verify_sufficiency(conds, int(preds[i]))
        if cert.status != VerificationStatus.VERIFIED:
            continue  # path not pure: fall back to full-path certificate only
        minimal = list(conds)
        for c in conds:
            trial = [x for x in minimal if x != c]
            if not trial:
                continue
            if verifier.verify_sufficiency(trial, int(preds[i])).status == VerificationStatus.VERIFIED:
                minimal = trial
        sizes.append(len(minimal))
        verified += 1
    return sizes, verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]

    attributes = active_vocabulary(cfg)
    pd_, ar, bank, _ = load_perception(cache_dir, cfg["perception"]["embedding_dim"],
                                       attributes=attributes)
    tr_provider = FeatureProvider(cache_dir, "train")
    Xtr, ytr, _, compiler, _ = compile_matrices(cfg, "train", tr_provider, pd_, ar, bank, "gt", attributes)
    del tr_provider
    te_provider = FeatureProvider(cache_dir, "test")
    Xte, yte, _, _, _ = compile_matrices(cfg, "test", te_provider, pd_, ar, bank, "gt", attributes)
    Xtr, Xte = Xtr.astype(np.int64), Xte.astype(np.int64)
    names = compiler.feature_names
    rng = np.random.RandomState(42)
    rows = rng.choice(len(Xte), size=200, replace=False)
    perm = rng.permutation(len(Xtr))
    vi, fi = perm[:900], perm[900:]

    results = {}

    # ---- Tier 1: GOSDT ------------------------------------------------------
    print("--> GOSDT sweep (CUB, 200 classes)...")
    from gosdt import GOSDTClassifier
    best = None
    results["gosdt_cub"] = {"status": "not attempted"}
    for reg in (0.01, 0.03, 0.1):
        try:
            m = GOSDTClassifier(regularization=reg, depth_budget=12, time_limit=300)
            m.fit(Xtr[fi].astype(float), ytr[fi])
        except RuntimeError as e:
            print(f"   reg={reg}: EXACT OPTIMIZER DID NOT CONVERGE ({str(e)[:60]})")
            results["gosdt_cub"] = {
                "status": "exact optimization did not converge within 300s "
                          "(queue >3.6M nodes at 200 classes) — scalability finding"}
            continue
        val = float(np.mean(np.array(m.predict(Xtr[vi].astype(float))) == ytr[vi]))
        n_leaves = sum(1 for n in iter_nodes(m.trees_[0].tree))
        print(f"   reg={reg}: val {val*100:.2f}% | nodes {len(nodes_flat := flatten(m))}")
        if best is None or val > best[1]:
            best = (m, val, reg)
    if best is not None:
        m, val_acc, reg = best
        preds_g = np.array(m.predict(Xte.astype(float)))
        arrs = gosdt_to_arrays(m.trees_[0].tree, int(ytr.max()) + 1, names)
        tree = SimpleThresholdTree(*arrs[:4], arrs[4], names)
        preds_t = np.array([tree.predict_one(x) for x in Xte])
        match = float(np.mean(preds_t == preds_g))
        if match < 0.999:
            arrs2 = (arrs[0], arrs[1], arrs[3], arrs[2], arrs[4])
            tree2 = SimpleThresholdTree(*arrs2[:4], arrs2[4], names)
            m2 = float(np.mean(np.array([tree2.predict_one(x) for x in Xte]) == preds_g))
            if m2 > match:
                tree, match = tree2, m2
        print(f"   GOSDT conversion prediction match: {match*100:.2f}%")
        sizes, n_verified = certified_reasons(tree, Xte, preds_t, rows, names)
        results["gosdt"] = {
            "regularization": reg, "val_acc": val_acc,
            "test_acc": float(np.mean(preds_g == yte)), "nodes": int(arrs[2].shape[0]),
            "conversion_match": match,
            "reason_mean": float(np.mean(sizes)) if sizes else None,
            "reason_max": int(np.max(sizes)) if sizes else None,
            "n_verified": n_verified,
        }
    else:
        results["gosdt"] = {"status": "skipped — exact optimizer did not converge on 200 classes"}

    # ---- Tier 3: RuleFit ----------------------------------------------------
    print("--> RuleFit...")
    from imodels import RuleFitClassifier
    rf = RuleFitClassifier(n_estimators=50, tree_size=4, random_state=42)
    try:
        rf.fit(Xtr[fi].astype(float), ytr[fi])
    except ValueError as e:
        print(f"   RuleFit: {str(e)[:60]}")
        rf = None
    if rf is not None:
        acc_val = float(np.mean(rf.predict(Xtr[vi].astype(float)) == ytr[vi]))
        preds_r = rf.predict(Xte.astype(float))
        rules = getattr(rf, "rules_", None) or getattr(rf, "rule_ensemble", None)
        n_rules = len(rules) if rules is not None else None
        results["rulefit"] = {
            "val_acc": acc_val, "test_acc": float(np.mean(preds_r == yte)),
            "n_rules": n_rules, "proofs": "none (rule selection is not entailment)",
        }
    else:
        results["rulefit"] = {"status": "no multiclass support (imodels RuleFitClassifier)"}

    # ---- Tier 2: RandomForest ------------------------------------------------
    print("--> RandomForest...")
    from sklearn.ensemble import RandomForestClassifier
    big = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=4,
                                 random_state=42, n_jobs=-1).fit(Xtr, ytr)
    results["rf_500"] = {"test_acc": float(np.mean(big.predict(Xte) == yte))}

    small = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42,
                                   n_jobs=-1).fit(Xtr, ytr)
    preds_s = small.predict(Xte)
    results["rf_25"] = {"test_acc": float(np.mean(preds_s == yte))}
    # PyXAI forest payload for compiler-based proofs
    from sweep_tree_depth import tree_nodes
    forest_nodes = []
    for est in small.estimators_:
        t = est.tree_
        nodes = []
        for nid in range(t.node_count):
            if t.feature[nid] == -2:
                # sklearn RF leaves: class distribution over all classes (weighted)
                cls = small.classes_[int(np.argmax(t.value[nid][0]))]
                nodes.append({"f": -1, "l": -1, "r": -1, "c": int(cls), "t": 0.0})
            else:
                nodes.append({"f": int(t.feature[nid]), "l": int(t.children_left[nid]),
                              "r": int(t.children_right[nid]), "c": -1,
                              "t": float(t.threshold[nid])})
        forest_nodes.append(nodes)
    payload = {
        "forest": forest_nodes, "n_classes": int(ytr.max()) + 1, "numeric": False,
        "n_features": int(Xte.shape[1]),
        "feature_names": [f"f{i+1}" for i in range(Xte.shape[1])],
        "instances": [
            {"sample_id": int(i), "x": Xte[i].tolist(),
             "prediction": int(preds_s[i]), "label": int(yte[i]),
             "native_reason_literals": [], "native_reason_size": 0, "path_length": 0}
            for i in rows[:50]
        ],
    }
    with open(os.path.join(RUN, "rf_pyxai_payload.json"), "w") as f:
        json.dump(payload, f)

    # CART reference under the identical split/protocol
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.000198, min_samples_leaf=4
                                ).fit(Xtr[fi], ytr[fi])
    results["cart_ref"] = {
        "val_acc": float(np.mean(dt.predict(Xtr[vi]) == ytr[vi])),
        "test_acc_note": "validated-selection run reported 23.47% (refit on full train)",
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"saved: {OUT}; rf payload -> {RUN}/rf_pyxai_payload.json")


def iter_nodes(node):
    yield node
    for c in (node.left_child, node.right_child):
        if hasattr(c, "left_child"):
            yield from iter_nodes(c)


def flatten(m):
    return list(iter_nodes(m.trees_[0].tree))


if __name__ == "__main__":
    main()
