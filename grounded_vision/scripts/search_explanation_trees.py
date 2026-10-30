"""
Script: search_explanation_trees.py
===================================
Explanation-aware tree search: instead of uniform depth limits, grow and prune
ASYMMETRIC trees and locate the accuracy-vs-certified-reason-size Pareto
frontier ("which tree posits the best explanation").

Families searched (all on the same compiled concept matrices, `--regions gt`):
  1. ccp    — cost-complexity pruning path of the full-depth tree (asymmetric subtrees)
  2. leaves — max_leaf_nodes budgets (asymmetric depth, leaf-limited)
  3. msl    — min_samples_leaf floors at unrestricted depth

Every candidate is scored on the fixed 200-instance test sample with the same
protocol as the depth sweep. Pareto frontier candidates are exported as PyXAI
payloads for the minimum-cardinality audit.
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
from sweep_tree_depth import tree_nodes  # noqa: E402
from grounded_vision.pipeline import load_perception, FeatureProvider  # noqa: E402
from grounded_vision.reasoning.tree_head import CertifiableTreeHead  # noqa: E402
from grounded_vision.explanations.pyxai_backend import PyXAIBackend  # noqa: E402


def wrap(fitted: DecisionTreeClassifier, feature_names) -> CertifiableTreeHead:
    head = CertifiableTreeHead(feature_names=feature_names)
    head.clf = fitted
    head.is_fitted = True
    return head


def evaluate(head, Xte, yte, rows):
    backend = PyXAIBackend(head)
    preds = head.predict(Xte)
    reason_sizes, path_lens = [], []
    con_avail, con_flips = 0, []
    instances = []
    for i in rows:
        x = Xte[i]
        pred = int(preds[i])
        exp = backend.explain_sufficient(x, pred, sample_id=int(i))
        literals = [-(c.feature_idx + 1) if c.operator == "<=" else (c.feature_idx + 1)
                    for c in exp.conditions]
        path_len = len(head.get_decision_path(x))
        reason_sizes.append(len(literals))
        path_lens.append(path_len)
        con = backend.explain_contrastive(x, pred, sample_id=int(i))
        if con is not None:
            con_avail += 1
            xc = x.copy()
            for c in con.conditions_to_alter:
                xc[c.feature_idx] = 1 - xc[c.feature_idx]
            con_flips.append(int(head.predict(xc.reshape(1, -1))[0]) != pred)
        instances.append({
            "sample_id": int(i), "x": x.tolist(), "prediction": pred,
            "label": int(yte[i]), "native_reason_literals": literals,
            "native_reason_size": len(literals), "path_length": path_len,
        })
    return {
        "reason_mean": float(np.mean(reason_sizes)),
        "reason_median": float(np.median(reason_sizes)),
        "reason_max": int(np.max(reason_sizes)),
        "path_mean": float(np.mean(path_lens)),
        "frac_reason_lt_path": float(np.mean(np.array(reason_sizes) < np.array(path_lens))),
        "contrastive_flip_rate": float(np.mean(con_flips)) if con_flips else None,
        "contrastive_available_rate": con_avail / len(rows),
        "instances": instances,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    parser.add_argument("--n-instances", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    embed_dim = cfg["perception"]["embedding_dim"]

    part_detector, attr_reader, bank, _ = load_perception(cache_dir, embed_dim)
    train_provider = FeatureProvider(cache_dir, "train")
    Xtr, ytr, _, compiler, _ = compile_matrices(cfg, "train", train_provider,
                                                part_detector, attr_reader, bank, "gt")
    del train_provider
    test_provider = FeatureProvider(cache_dir, "test")
    Xte, yte, _, _, _ = compile_matrices(cfg, "test", test_provider,
                                         part_detector, attr_reader, bank, "gt")
    Xtr, Xte = Xtr.astype(np.int64), Xte.astype(np.int64)

    out_dir = os.path.join(cache_dir, "tree_search")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    rows = rng.choice(len(Xte), size=min(args.n_instances, len(Xte)), replace=False)

    # ---- Build candidate family ------------------------------------------
    full = DecisionTreeClassifier(max_depth=None, min_samples_leaf=4, random_state=42)
    full.fit(Xtr, ytr)
    path = full.cost_complexity_pruning_path(Xtr, ytr)
    alphas = np.unique(path.ccp_alphas[path.ccp_alphas > 0])
    alphas = alphas[np.linspace(0, len(alphas) - 1, 12).astype(int)]

    candidates = []
    for a in alphas:
        candidates.append((f"ccp_{a:.5f}", {"ccp_alpha": float(a), "min_samples_leaf": 4}))
    for nl in [8, 16, 32, 64, 128, 256, 512, 1024]:
        candidates.append((f"leaves_{nl}", {"max_leaf_nodes": nl, "min_samples_leaf": 4}))
    for msl in [1, 2, 8, 16, 32, 64]:
        candidates.append((f"msl_{msl}", {"min_samples_leaf": msl}))

    results = []
    for name, params in candidates:
        dt = DecisionTreeClassifier(random_state=42, **params)
        dt.fit(Xtr, ytr)
        head = wrap(dt, compiler.feature_names)
        m = evaluate(head, Xte, yte, rows)
        entry = {
            "name": name, "params": params,
            "n_leaves": int(dt.get_n_leaves()),
            "train_acc": float(np.mean(dt.predict(Xtr) == ytr)),
            "test_acc": float(np.mean(dt.predict(Xte) == yte)),
            **{k: v for k, v in m.items() if k != "instances"},
        }
        entry["_instances"] = m["instances"]
        entry["_nodes"] = tree_nodes(head)
        results.append(entry)
        print(f"{name:16s} | leaves {entry['n_leaves']:5d} | test {entry['test_acc']*100:5.2f}% "
              f"| reason {entry['reason_mean']:5.2f} (med {entry['reason_median']:.0f}, "
              f"max {entry['reason_max']:2d}) | path {entry['path_mean']:5.2f} "
              f"| compr {entry['frac_reason_lt_path']*100:4.1f}%")

    # ---- Pareto frontier (min reason size, max test acc) --------------------
    def dominates(a, b):
        return (a["test_acc"] >= b["test_acc"] and a["reason_mean"] <= b["reason_mean"]
                and (a["test_acc"] > b["test_acc"] or a["reason_mean"] < b["reason_mean"]))

    frontier = [r for r in results if not any(dominates(o, r) for o in results)]
    frontier.sort(key=lambda r: r["reason_mean"])

    # Merge the depth-sweep curve for reporting
    sweep_path = os.path.join(cache_dir, "depth_sweep", "depth_sweep_summary.json")
    sweep = []
    if os.path.exists(sweep_path):
        for e in json.load(open(sweep_path)):
            sweep.append({"name": f"depth_{e['depth']}", "n_leaves": e["n_leaves"],
                          "test_acc": e["test_acc"], "reason_mean": e["mean_native_reason_size"],
                          "reason_median": e["mean_native_reason_size"],
                          "reason_max": e["max_native_reason_size"], "path_mean": e["mean_path_length"]})

    print("\nPARETO FRONTIER (search families; depth curve merged for reference):")
    merged = results + sweep
    frontier_all = [r for r in merged if not any(dominates(o, r) for o in merged)]
    for r in frontier_all:
        tag = "SEARCH" if "_instances" in r else "depth"
        print(f"  [{tag}] {r['name']:16s} | test {r['test_acc']*100:5.2f}% | "
              f"reason {r['reason_mean']:5.2f} | leaves {r.get('n_leaves')}")

    for r in frontier_all:
        if "_instances" not in r:
            continue
        safe = r["name"].replace(".", "p")
        with open(os.path.join(out_dir, f"frontier_{safe}.json"), "w") as f:
            json.dump({"n_features": int(Xte.shape[1]),
                       "feature_names": compiler.feature_names,
                       "nodes": r["_nodes"], "instances": r["_instances"],
                       "meta": {k: v for k, v in r.items() if not k.startswith("_")}}, f)

    for r in results:
        r.pop("_instances", None)
        r.pop("_nodes", None)
    with open(os.path.join(out_dir, "tree_search_summary.json"), "w") as f:
        json.dump({"results": results,
                   "frontier": [{k: v for k, v in r.items()} for r in frontier_all]}, f, indent=2)
    print(f"\nsaved: {os.path.join(out_dir, 'tree_search_summary.json')}")


if __name__ == "__main__":
    main()
