"""
Script: compare_tree_methods_binary.py
======================================
Full tree-based head comparison on a BINARY concept target (CelebA `Male`
from 39 concepts) — the setting where GOSDT (optimal sparse trees) and
RuleFit are natively supported. All methods share the identical split and
protocol; GOSDT trees are converted to plain threshold trees (gated by an
exact-prediction check) so NATIVE proofs apply unchanged.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_tree_methods import SimpleThresholdTree, gosdt_to_arrays, certified_reasons
from analyze_full_run import compile_matrices
from grounded_vision.pipeline import load_perception, FeatureProvider, active_vocabulary

OUT = "/Users/cril/tanmoy/research/data/grounded_vision_runs/celeba_concepts/tree_methods_comparison.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/celeba_concepts.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]

    attributes = active_vocabulary(cfg)
    pd_, ar, bank, _ = load_perception(cache_dir, cfg["perception"]["embedding_dim"],
                                       attributes=attributes)
    tr = FeatureProvider(cache_dir, "train")
    Xtr, ytr, _, compiler, _ = compile_matrices(cfg, "train", tr, pd_, ar, bank, "gt", attributes)
    del tr
    te = FeatureProvider(cache_dir, "test")
    Xte, yte, _, _, _ = compile_matrices(cfg, "test", te, pd_, ar, bank, "gt", attributes)
    Xtr, Xte = Xtr.astype(np.int64), Xte.astype(np.int64)
    names = compiler.feature_names

    rng = np.random.RandomState(42)
    rows = rng.choice(len(Xte), size=200, replace=False)
    perm = rng.permutation(len(Xtr))
    vi, fi = perm[:3000], perm[3000:]
    results = {}

    # ---- CART reference (validated selection on this split) ------------------
    from sklearn.tree import DecisionTreeClassifier
    best = None
    for name, params in [("d4", {"max_depth": 4}), ("d8", {"max_depth": 8}),
                         ("leaves_64", {"max_leaf_nodes": 64}),
                         ("leaves_256", {"max_leaf_nodes": 256}),
                         ("msl_32", {"min_samples_leaf": 32})]:
        dt = DecisionTreeClassifier(random_state=42, **params).fit(Xtr[fi], ytr[fi])
        val = float(np.mean(dt.predict(Xtr[vi]) == ytr[vi]))
        if best is None or val > best[2]:
            best = (name, params, val)
    cart = DecisionTreeClassifier(random_state=42, **best[1]).fit(Xtr, ytr)
    preds_c = cart.predict(Xte)
    # native proofs via CertifiableTreeHead machinery
    from grounded_vision.reasoning.tree_head import CertifiableTreeHead
    from grounded_vision.explanations.pyxai_backend import PyXAIBackend
    head = CertifiableTreeHead(feature_names=names, class_names=["no", "yes"])
    head.clf, head.is_fitted = cart, True
    backend = PyXAIBackend(head)
    sizes = [backend.explain_sufficient(Xte[i], int(preds_c[i]), sample_id=int(i)).sparsity
             for i in rows]
    results["cart"] = {"selected": best[0], "val_acc": best[2],
                       "test_acc": float(np.mean(preds_c == yte)),
                       "reason_mean": float(np.mean(sizes)), "proofs": "native"}

    # ---- GOSDT ----------------------------------------------------------------
    print("--> GOSDT (binary)...")
    from gosdt import GOSDTClassifier
    gbest = None
    for reg in (0.005, 0.02, 0.05, 0.1):
        m = GOSDTClassifier(regularization=reg, time_limit=300)
        try:
            m.fit(Xtr[fi].astype(float), ytr[fi])
        except RuntimeError as e:
            print(f"   reg={reg}: no convergence ({str(e)[:50]})")
            continue
        val = float(np.mean(np.array(m.predict(Xtr[vi].astype(float))) == ytr[vi]))
        print(f"   reg={reg}: val {val*100:.2f}%")
        if gbest is None or val > gbest[1]:
            gbest = (m, val, reg)
    if gbest is not None:
        m, val_acc, reg = gbest
        preds_g = np.array(m.predict(Xte.astype(float)))
        arrs = gosdt_to_arrays(m.trees_[0].tree, 2, names)
        tree = SimpleThresholdTree(*arrs[:4], arrs[4], names)
        match = float(np.mean(np.array([tree.predict_one(x) for x in Xte]) == preds_g))
        if match < 0.999:  # child-convention fallback
            arrs2 = (arrs[0], arrs[1], arrs[3], arrs[2], arrs[4])
            t2 = SimpleThresholdTree(*arrs2[:4], arrs2[4], names)
            m2 = float(np.mean(np.array([t2.predict_one(x) for x in Xte]) == preds_g))
            if m2 > match:
                tree, match = t2, m2
        print(f"   conversion prediction match: {match*100:.2f}%")
        sizes_g, n_ver = certified_reasons(tree, Xte, preds_g, rows, names)
        results["gosdt"] = {
            "regularization": reg, "val_acc": val_acc,
            "test_acc": float(np.mean(preds_g == yte)),
            "nodes": int(arrs[2].shape[0]), "conversion_match": match,
            "reason_mean": float(np.mean(sizes_g)) if sizes_g else None,
            "n_verified": n_ver, "proofs": "native (converted tree)",
        }
    else:
        results["gosdt"] = {"status": "did not converge"}

    # ---- RuleFit ---------------------------------------------------------------
    print("--> RuleFit (binary)...")
    from imodels import RuleFitClassifier
    rf = RuleFitClassifier(n_estimators=50, tree_size=4, random_state=42,
                           max_rules=60)
    rf.fit(Xtr[fi].astype(float), ytr[fi])
    preds_r = rf.predict(Xte.astype(float))
    n_rules = len(getattr(rf, "rules_", []) or [])
    results["rulefit"] = {
        "val_acc": float(np.mean(rf.predict(Xtr[vi].astype(float)) == ytr[vi])),
        "test_acc": float(np.mean(preds_r == yte)), "n_rules": n_rules,
        "proofs": "none (rule selection is not entailment)",
    }

    # ---- RandomForest ----------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=4,
                                    random_state=42, n_jobs=-1).fit(Xtr, ytr)
    results["rf_500"] = {"test_acc": float(np.mean(forest.predict(Xte) == yte)),
                         "proofs": "compiler-based only (PyXAI), not native"}

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
