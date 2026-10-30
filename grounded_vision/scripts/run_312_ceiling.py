"""
Script: run_312_ceiling.py
==========================
Closes the "does the 28-attribute ceiling rise with the full 312?" question.

Phase 1 (semantic ceiling): decision tree + logistic regression on the FULL
312 ground-truth binary CUB attributes — no perception involved. Reference
points from the 28-attribute canonical set: GT-concepts tree 16.71%,
logreg 23.85%.

Phase 2 (realizable ceiling): one balanced logistic probe per attribute on
pooled GT part-region DINO embeddings (part chosen by attribute-name prefix;
whole-bird attributes read from the CLS token; wing/leg attributes use the
mean of left/right embeddings). Tree + logreg on the 312 PREDICTED binaries.

The full predicate/grounding/compiler integration over 312 attributes is
future work; this experiment measures the ceiling question directly.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_prototype_tree import pool_part_embeddings  # noqa: E402
from grounded_vision.pipeline import FeatureProvider  # noqa: E402

PREFIX_PART = [
    ("back_of_neck", "back"), ("upper_tail", "tail"), ("lower_tail", "tail"),
    ("bill", "beak"), ("crown", "crown"), ("forehead", "forehead"),
    ("nape", "nape"), ("throat", "throat"), ("breast", "breast"),
    ("belly", "belly"), ("back", "back"), ("wing", "wing"),
    ("tail", "tail"), ("leg", "leg"), ("foot", "leg"),
    ("head", "crown"), ("eye", "crown"),          # approximate bindings
    ("upperparts", "back"), ("underparts", "belly"),
    ("size", "whole"), ("shape", "whole"), ("plumage", "whole"),
]


def attribute_region(name_312: str) -> str:
    stem = name_312.removeprefix("has_").split("::")[0]
    for prefix, region in PREFIX_PART:
        if stem.startswith(prefix):
            return region
    return "whole"


def load_full_attributes(data_dir):
    """{img_id -> np.array[312] float32} for all CUB attributes."""
    with open(os.path.join(data_dir, "attributes.txt")) as f:
        attr_names = [line.strip().split(None, 1)[1] for line in f if line.strip()]
    n_attrs = len(attr_names)
    labels = {}
    with open(os.path.join(data_dir, "attributes", "image_attribute_labels.txt")) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 3:
                continue
            img_id, attr_id, present = int(fields[0]), int(fields[1]), int(fields[2])
            vec = labels.get(img_id)
            if vec is None:
                vec = np.zeros(n_attrs, dtype=np.float32)
                labels[img_id] = vec
            vec[attr_id - 1] = 1.0 if present else 0.0
    return attr_names, labels


def region_vector(emb, cls, region):
    if region == "whole":
        return cls
    if region == "wing":
        parts = [emb[p] for p in ("left_wing", "right_wing") if p in emb]
    elif region == "leg":
        parts = [emb[p] for p in ("left_leg", "right_leg") if p in emb]
    else:
        parts = [emb[region]] if region in emb else []
    return np.mean(parts, axis=0) if parts else None


def heads_table(Xtr, ytr, Xte, yte):
    tree = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4, random_state=42).fit(Xtr, ytr)
    logreg = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    return {
        "tree_d12": {
            "train_acc": float(np.mean(tree.predict(Xtr) == ytr)),
            "test_acc": float(np.mean(tree.predict(Xte) == yte)),
        },
        "logreg_C1": {
            "train_acc": float(np.mean(logreg.predict(Xtr) == ytr)),
            "test_acc": float(np.mean(logreg.predict(Xte) == yte)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    data_dir = cfg["data"]["data_dir"]

    attr_names, gt_by_id = load_full_attributes(data_dir)
    regions = [attribute_region(n) for n in attr_names]
    from collections import Counter
    print(f"--> {len(attr_names)} attributes; region binding: "
          f"{dict(Counter(regions))}")

    # ---- Phase 1: GT 312 labels -------------------------------------------
    tr_ids, ytr, pooled_tr = pool_part_embeddings(cfg, "train", FeatureProvider(cache_dir, "train"))
    te_ids, yte, pooled_te = pool_part_embeddings(cfg, "test", FeatureProvider(cache_dir, "test"))
    GTtr = np.stack([gt_by_id[s] for s in tr_ids])
    GTte = np.stack([gt_by_id[s] for s in te_ids])
    phase1 = heads_table(GTtr, ytr, GTte, yte)

    # ---- Phase 2: probes on grounded region embeddings --------------------
    d_tr = np.load(os.path.join(cache_dir, "train_features.npz"))
    cls_tr = {int(s): v for s, v in zip(d_tr["sample_ids"], d_tr["cls_tokens"])}
    d_te = np.load(os.path.join(cache_dir, "test_features.npz"))
    cls_te = {int(s): v for s, v in zip(d_te["sample_ids"], d_te["cls_tokens"])}

    dim = len(next(iter(cls_tr.values())))

    def region_matrices(pooled, ids, cls_map):
        """{region: [N, dim]} with NaN rows where the region is unavailable."""
        mats = {r: np.full((len(ids), dim), np.nan, dtype=np.float32)
                for r in set(regions)}
        for i, sid in enumerate(ids):
            emb, cls = pooled[sid], cls_map[sid]
            for r in mats:
                v = region_vector(emb, cls, r)
                if v is not None:
                    mats[r][i] = v
        return mats

    mats_tr = region_matrices(pooled_tr, tr_ids, cls_tr)
    mats_te = region_matrices(pooled_te, te_ids, cls_te)

    preds_tr = np.zeros_like(GTtr)
    preds_te = np.zeros_like(GTte)
    probe_acc = []
    for a, region in enumerate(regions):
        Ftr_a, Fte_a = mats_tr[region], mats_te[region]
        mask_tr = ~np.isnan(Ftr_a[:, 0])
        mask_te = ~np.isnan(Fte_a[:, 0])
        if mask_tr.sum() < 20 or len(np.unique(GTtr[mask_tr, a])) < 2:
            continue
        probe = LogisticRegression(max_iter=200, class_weight="balanced").fit(
            Ftr_a[mask_tr], GTtr[mask_tr, a])
        preds_tr[mask_tr, a] = probe.predict(Ftr_a[mask_tr])
        if mask_te.sum():
            preds_te[mask_te, a] = probe.predict(Fte_a[mask_te])
            probe_acc.append(float(np.mean(preds_te[mask_te, a] == GTte[mask_te, a])))

    phase2 = heads_table(preds_tr, ytr, preds_te, yte)
    valid_acc = [x for x in probe_acc if x is not None]
    report = {
        "phase1_gt_312": phase1,
        "phase2_probe_312": phase2,
        "reference_28_gt": {"tree_d12": {"test_acc": 0.1671}, "logreg_C1": {"test_acc": 0.2385}},
        "reference_28_predicted": {"tree_d12": {"test_acc": 0.2268}, "logreg_C1": {"test_acc": 0.2994}},
        "mean_probe_attribute_accuracy": float(np.mean(valid_acc)),
        "n_probes_trained": len(valid_acc),
        "concept_agreement_pred_vs_gt": float(np.mean(preds_tr == GTtr)),
    }
    out = os.path.join(cache_dir, "results", "ceiling_312.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
