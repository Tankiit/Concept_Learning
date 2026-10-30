"""
Script: sweep_tree_depth.py
===========================
Maps the accuracy-vs-certified-reason-size trade-off for the concept tree:
fits CART heads at several depths on the SAME compiled concept matrices
(GT regions), computes native sufficient reasons, path lengths, and
contrastives on a fixed test sample, and emits one PyXAI payload per depth
for the minimum-cardinality audit (see run_pyxai_modal.py).
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import pickle
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_full_run import compile_matrices  # noqa: E402
from grounded_vision.pipeline import load_perception, FeatureProvider  # noqa: E402
from grounded_vision.reasoning.tree_head import CertifiableTreeHead  # noqa: E402
from grounded_vision.explanations.pyxai_backend import PyXAIBackend  # noqa: E402


def tree_nodes(tree_head):
    tree = tree_head.clf.tree_
    nodes = []
    for nid in range(tree.node_count):
        if tree.feature[nid] == -2:
            nodes.append({"f": -1, "l": -1, "r": -1, "c": int(np.argmax(tree.value[nid][0]))})
        else:
            nodes.append({"f": int(tree.feature[nid]), "l": int(tree.children_left[nid]),
                          "r": int(tree.children_right[nid]), "c": -1})
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    parser.add_argument("--depths", type=str, default="2,3,4,5,6,8,10,12")
    parser.add_argument("--n-instances", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    leaf = cfg["reasoning"]["min_samples_leaf"]
    threshold = cfg["predicates"].get("probability_threshold", 0.5)
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

    out_dir = os.path.join(cache_dir, "depth_sweep")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(42)
    rows = rng.choice(len(Xte), size=min(args.n_instances, len(Xte)), replace=False)

    summary = []
    for depth in [int(d) for d in args.depths.split(",")]:
        head = CertifiableTreeHead(max_depth=depth, min_samples_leaf=leaf,
                                   feature_names=compiler.feature_names)
        head.fit(Xtr, ytr)
        train_acc = float(np.mean(head.predict(Xtr) == ytr))
        test_acc = float(np.mean(head.predict(Xte) == yte))

        backend = PyXAIBackend(head)
        instances, reason_sizes, path_lens, ratios = [], [], [], []
        con_avail, con_sizes, con_flips = 0, [], []
        for i in rows:
            x = Xte[i]
            pred = int(head.predict(x.reshape(1, -1))[0])
            exp = backend.explain_sufficient(x, pred, sample_id=int(i))
            literals = [-(c.feature_idx + 1) if c.operator == "<=" else (c.feature_idx + 1)
                        for c in exp.conditions]
            path_len = len(head.get_decision_path(x))
            reason_sizes.append(len(literals))
            path_lens.append(path_len)
            ratios.append(len(literals) / max(1, path_len))
            con = backend.explain_contrastive(x, pred, sample_id=int(i))
            if con is not None:
                con_avail += 1
                con_sizes.append(len(con.conditions_to_alter))
                xc = x.copy()
                for c in con.conditions_to_alter:
                    xc[c.feature_idx] = 1 - xc[c.feature_idx]
                con_flips.append(int(head.predict(xc.reshape(1, -1))[0]) != pred)
            instances.append({
                "sample_id": int(i), "x": x.tolist(), "prediction": pred,
                "label": int(yte[i]), "native_reason_literals": literals,
                "native_reason_size": len(literals), "path_length": path_len,
            })

        entry = {
            "depth": depth,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_leaves": int(head.clf.get_n_leaves()),
            "mean_path_length": float(np.mean(path_lens)),
            "mean_native_reason_size": float(np.mean(reason_sizes)),
            "max_native_reason_size": int(np.max(reason_sizes)),
            "mean_ratio": float(np.mean(ratios)),
            "contrastive_available_rate": con_avail / len(rows),
            "mean_contrastive_conditions": float(np.mean(con_sizes)) if con_sizes else None,
            "contrastive_flip_rate": float(np.mean(con_flips)) if con_flips else None,
        }
        summary.append(entry)
        print(f"d={depth:2d} | leaves {entry['n_leaves']:4d} | train {train_acc*100:5.2f}% "
              f"| test {test_acc*100:5.2f}% | reason {entry['mean_native_reason_size']:.2f} "
              f"(max {entry['max_native_reason_size']}) | path {entry['mean_path_length']:.2f} "
              f"| contrast {entry['mean_contrastive_conditions']}"
              if entry["mean_contrastive_conditions"] is not None else f"d={depth}: no contrastives")

        with open(os.path.join(out_dir, f"depth_d{depth}.json"), "w") as f:
            json.dump({"n_features": int(head.clf.n_features_in_),
                       "feature_names": compiler.feature_names,
                       "nodes": tree_nodes(head),
                       "instances": instances}, f)

    with open(os.path.join(out_dir, "depth_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {os.path.join(out_dir, 'depth_sweep_summary.json')}")


if __name__ == "__main__":
    main()
