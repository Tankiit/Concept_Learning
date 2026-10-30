"""
Script: export_pyxai_comparison.py
==================================
Exports the fitted concept tree plus sampled test instances and their
NATIVE (greedy, irreducible) sufficient reasons to JSON, for a PyXAI
minimum-cardinality comparison on a Linux machine (PyXAI's bundled d4
compiler is an x86-64 Linux binary and cannot run on macOS).
"""

from __future__ import annotations
import json
import argparse
import pickle
import numpy as np
import yaml

from grounded_vision.pipeline import build_dataset, load_perception, perceive_sample, FeatureProvider, active_vocabulary
from grounded_vision.explanations.pyxai_backend import PyXAIBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    parser.add_argument("--n-instances", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    threshold = cfg["predicates"].get("probability_threshold", 0.5)
    attributes = active_vocabulary(cfg)

    with open(f"{cache_dir}/tree_head.pkl", "rb") as f:
        tree_head = pickle.load(f)["tree_head"]
    tree = tree_head.clf.tree_

    # Flatten the sklearn tree: leaves carry the argmax class id
    nodes = []
    for nid in range(tree.node_count):
        if tree.feature[nid] == -2:
            nodes.append({"f": -1, "l": -1, "r": -1, "c": int(np.argmax(tree.value[nid][0]))})
        else:
            nodes.append({"f": int(tree.feature[nid]), "l": int(tree.children_left[nid]),
                          "r": int(tree.children_right[nid]), "c": -1})

    part_detector, attr_reader, bank, _ = load_perception(cache_dir, cfg["perception"]["embedding_dim"], attributes=attributes)
    provider = FeatureProvider(cache_dir, "test")
    dataset = build_dataset(cfg, "test")

    rng = np.random.RandomState(42)
    sampled = set(rng.choice(len(dataset), size=min(args.n_instances, len(dataset)), replace=False).tolist())

    backend = PyXAIBackend(tree_head)
    instances = []
    for i in range(len(dataset)):
        if i not in sampled:
            continue
        sample = dataset[i]
        if not provider.has_cache or sample.sample_id not in provider.index:
            continue
        result = perceive_sample(sample, provider, part_detector, attr_reader, bank,
                                 region_mode="gt", probability_threshold=threshold,
                                 attributes=attributes)
        x = np.zeros(tree_head.clf.n_features_in_, dtype=np.int64)
        from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
        compiler = FeatureCompiler(mode=RepresentationMode(cfg["experiment"]["representation"]), attributes=attributes)
        x = compiler.compile_database(result.db, use_probabilities=False).astype(np.int64)
        pred = int(tree_head.predict(x.reshape(1, -1))[0])
        exp = backend.explain_sufficient(x, pred, sample_id=int(sample.sample_id))
        path_len = len(tree_head.get_decision_path(x))
        literals = [-(c.feature_idx + 1) if c.operator == "<=" else (c.feature_idx + 1)
                    for c in exp.conditions]
        instances.append({
            "sample_id": int(sample.sample_id),
            "x": x.tolist(),
            "prediction": pred,
            "label": int(sample.label),
            "native_reason_literals": literals,
            "native_reason_size": len(literals),
            "path_length": path_len,
        })

    payload = {
        "n_features": int(tree_head.clf.n_features_in_),
        "feature_names": tree_head.feature_names,
        "nodes": nodes,
        "instances": instances,
    }
    out = args.out or f"{cache_dir}/pyxai_export_n{len(instances)}.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    sizes = [inst["native_reason_size"] for inst in instances]
    print(f"Exported {len(instances)} instances; native reason size mean {np.mean(sizes):.2f} "
          f"min {min(sizes)} max {max(sizes)} -> {out}")


if __name__ == "__main__":
    main()
