"""
Script: rerun_tree_search_validated.py
======================================
Closes the methodological caveat of search_explanation_trees.py: candidates
are SELECTED on a held-out validation split carved from train, and the chosen
configuration is refit on the full train split and reported ONCE on test.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import yaml
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_full_run import compile_matrices  # noqa: E402
from search_explanation_trees import evaluate  # noqa: E402
from grounded_vision.pipeline import load_perception, FeatureProvider, active_vocabulary  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--n-instances", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    embed_dim = cfg["perception"]["embedding_dim"]

    attributes = active_vocabulary(cfg)
    part_detector, attr_reader, bank, _ = load_perception(cache_dir, embed_dim, attributes=attributes)
    train_provider = FeatureProvider(cache_dir, "train")
    Xtr, ytr, _, compiler, _ = compile_matrices(cfg, "train", train_provider,
                                                part_detector, attr_reader, bank, "gt", attributes)
    del train_provider
    test_provider = FeatureProvider(cache_dir, "test")
    Xte, yte, _, _, _ = compile_matrices(cfg, "test", test_provider,
                                         part_detector, attr_reader, bank, "gt", attributes)
    Xtr, Xte = Xtr.astype(np.int64), Xte.astype(np.int64)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(Xtr))
    n_val = int(args.val_fraction * len(Xtr))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    print(f"--> selection: fit on {len(fit_idx)}, validate on {n_val}")

    candidates = []
    full = DecisionTreeClassifier(min_samples_leaf=4, random_state=42).fit(Xtr[fit_idx], ytr[fit_idx])
    path = full.cost_complexity_pruning_path(Xtr[fit_idx], ytr[fit_idx])
    alphas = np.unique(path.ccp_alphas[path.ccp_alphas > 0])
    for a in alphas[np.linspace(0, len(alphas) - 1, 12).astype(int)]:
        candidates.append((f"ccp_{a:.5f}", {"ccp_alpha": float(a), "min_samples_leaf": 4}))
    for nl in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        candidates.append((f"leaves_{nl}", {"max_leaf_nodes": nl, "min_samples_leaf": 4}))
    for msl in [1, 2, 8, 16, 32, 64]:
        candidates.append((f"msl_{msl}", {"min_samples_leaf": msl}))

    rows = rng.choice(len(Xte), size=min(args.n_instances, len(Xte)), replace=False)
    val_scores = []
    for name, params in candidates:
        dt = DecisionTreeClassifier(random_state=42, **params).fit(Xtr[fit_idx], ytr[fit_idx])
        val_scores.append({
            "name": name, "params": params,
            "val_acc": float(np.mean(dt.predict(Xtr[val_idx]) == ytr[val_idx])),
            "leaves": int(dt.get_n_leaves()),
        })
    best = max(val_scores, key=lambda s: (round(s["val_acc"], 4), -s["leaves"]))
    print(f"--> selected on validation: {best['name']} (val acc {best['val_acc']*100:.2f}%)")

    # Refit the selected config on the FULL train split; report once on test.
    dt = DecisionTreeClassifier(random_state=42, **best["params"]).fit(Xtr, ytr)
    from sweep_tree_depth import tree_nodes
    from grounded_vision.reasoning.tree_head import CertifiableTreeHead
    head = CertifiableTreeHead(feature_names=compiler.feature_names)
    head.clf, head.is_fitted = dt, True
    m = evaluate(head, Xte, yte, rows)

    # Persist as the run's deployed tree (format matches train_head.py)
    import pickle
    from grounded_vision.pipeline import load_class_names
    class_names = load_class_names(cfg["data"]["data_dir"]) or list(range(cfg["data"]["num_classes"]))
    head.class_names = class_names
    with open(os.path.join(cache_dir, "tree_head.pkl"), "wb") as f:
        pickle.dump({"tree_head": head, "compiler": compiler,
                     "feature_names": compiler.feature_names,
                     "class_names": class_names, "region_mode": "gt",
                     "selection": best}, f)
    report = {
        "selection": "validation-split (15% of train), refit on full train, reported once on test",
        "selected": best,
        "candidates_scored": len(candidates),
        "n_leaves": int(dt.get_n_leaves()),
        "train_acc": float(np.mean(dt.predict(Xtr) == ytr)),
        "test_acc": float(np.mean(dt.predict(Xte) == yte)),
        **{k: v for k, v in m.items() if k != "instances"},
        "prior_test_selected_number_for_reference": 0.2380,
    }
    out = os.path.join(cache_dir, "tree_search", "validated_selection.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "candidates_scored"}, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
