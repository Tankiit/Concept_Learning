"""
Script: run_exemplar_leaves.py
==============================
Exemplar-grounded leaves: evidence layer on a chosen concept tree. Prediction
stays the leaf's (static) class so certificates are untouched; for each test
instance we retrieve the nearest training images that share its leaf
(case-based support: "predicted X; nearest certified cases are ...").

Metrics: leaf purity, top-1/top-5 same-leaf exemplar agreement with the leaf
class, split by correct/incorrect predictions.
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
from grounded_vision.pipeline import load_perception, FeatureProvider, load_class_names  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    parser.add_argument("--tree", type=str, default="ccp:0.00023",
                        help="'ccp:<alpha>' or 'leaves:<n>' concept-tree config")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    embed_dim = cfg["perception"]["embedding_dim"]

    part_detector, attr_reader, bank, _ = load_perception(cache_dir, embed_dim)
    train_provider = FeatureProvider(cache_dir, "train")
    Xtr, ytr, _, compiler, _ = compile_matrices(cfg, "train", train_provider,
                                                part_detector, attr_reader, bank, "gt")
    train_ids = [int(s) for s in train_provider.sample_ids]
    del train_provider
    test_provider = FeatureProvider(cache_dir, "test")
    Xte, yte, _, _, _ = compile_matrices(cfg, "test", test_provider,
                                         part_detector, attr_reader, bank, "gt")
    test_ids = [int(s) for s in test_provider.sample_ids]

    fam, val = args.tree.split(":")
    params = ({"ccp_alpha": float(val), "min_samples_leaf": 4} if fam == "ccp"
              else {"max_leaf_nodes": int(val), "min_samples_leaf": 4})
    dt = DecisionTreeClassifier(random_state=42, **params)
    dt.fit(Xtr, ytr)

    leaf_tr = dt.apply(Xtr)
    leaf_te = dt.apply(Xte)
    preds = dt.predict(Xte)
    test_acc = float(np.mean(preds == yte))

    # CLS-token cosine for retrieval, aligned to the compiled matrices
    d_tr = np.load(os.path.join(cache_dir, "train_features.npz"))
    d_te = np.load(os.path.join(cache_dir, "test_features.npz"))
    cls_tr = {int(s): v / max(1e-8, np.linalg.norm(v)) for s, v in zip(d_tr["sample_ids"], d_tr["cls_tokens"])}
    cls_te = {int(s): v / max(1e-8, np.linalg.norm(v)) for s, v in zip(d_te["sample_ids"], d_te["cls_tokens"])}

    by_leaf = {}
    for i, lf in enumerate(leaf_tr):
        by_leaf.setdefault(int(lf), []).append(i)

    class_names = load_class_names(cfg["data"]["data_dir"]) or []
    top1_hits, top5_hits = [], []
    by_correct = {True: [], False: []}
    examples = []
    rng = np.random.RandomState(42)
    example_rows = set(rng.choice(len(Xte), size=6, replace=False).tolist())
    for i in range(len(Xte)):
        lf = int(leaf_te[i])
        members = by_leaf.get(lf, [])
        if not members:
            continue
        sims = np.array([cls_te[test_ids[i]] @ cls_tr[train_ids[j]] for j in members])
        order = np.argsort(-sims)[: args.k]          # positions into `members`
        neigh = [members[j] for j in order]           # member ids
        neigh_sims = [float(sims[j]) for j in order]
        neigh_labels = [int(ytr[j]) for j in neigh]
        pred_i = int(preds[i])
        top1_hits.append(neigh_labels[0] == pred_i)
        top5_hits.append(any(l == pred_i for l in neigh_labels))
        correct = pred_i == int(yte[i])
        by_correct[correct].append(neigh_labels[0] == pred_i)
        if i in example_rows and len(examples) < 6:
            examples.append({
                "test_sample": test_ids[i],
                "predicted": class_names[pred_i] if pred_i < len(class_names) else str(pred_i),
                "true": class_names[int(yte[i])] if int(yte[i]) < len(class_names) else str(yte[i]),
                "correct": bool(correct),
                "nearest_same_leaf_exemplars": [
                    {"sample_id": train_ids[j],
                     "species": class_names[int(ytr[j])] if int(ytr[j]) < len(class_names) else str(ytr[j]),
                     "cosine": round(s, 3)}
                    for j, s in zip(neigh[:3], neigh_sims[:3])
                ],
            })

    leaf_sizes = np.array([len(by_leaf.get(int(l), [])) for l in leaf_te])
    report = {
        "tree": args.tree, "params": params,
        "leaves": int(dt.get_n_leaves()),
        "test_acc": test_acc,
        "top1_exemplar_matches_prediction": float(np.mean(top1_hits)),
        "topk_exemplar_matches_prediction": float(np.mean(top5_hits)),
        "top1_by_prediction_correctness": {
            "correct": float(np.mean(by_correct[True])) if by_correct[True] else None,
            "incorrect": float(np.mean(by_correct[False])) if by_correct[False] else None,
        },
        "mean_leaf_train_size": float(np.mean(leaf_sizes)),
        "examples": examples,
    }
    out = os.path.join(cache_dir, "exemplar_leaves_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    for e in examples[:3]:
        tag = "OK " if e["correct"] else "ERR"
        print(f"[{tag}] test {e['test_sample']}: pred {e['predicted']} (true {e['true']}) "
              f"| nearest certified cases: " +
              ", ".join(f"{x['species']} ({x['cosine']})" for x in e["nearest_same_leaf_exemplars"]))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
