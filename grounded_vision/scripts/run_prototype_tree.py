"""
Script: run_prototype_tree.py
=============================
Certifiable prototype-threshold tree: internal tests are
"cosine(image part embedding, prototype_k) >= tau" instead of concept bits.
Features come from the FITTED prototype bank (only prototypes backed by real
positive exemplars); any axis-aligned threshold tree is certifiable by the
existing verifier, so literals read "resembles prototype #k above tau" with
the same formal guarantees as the concept tree. Accuracy is NOT comparable to
the 36-concept rows (richer feature space) — this is the prototype track.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import yaml
import torch
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_full_run import compile_matrices  # noqa: E402
from sweep_tree_depth import tree_nodes  # noqa: E402
from grounded_vision.data.cub import CANONICAL_BOUND_ATTRIBUTES  # noqa: E402
from grounded_vision.pipeline import build_dataset, load_perception, FeatureProvider  # noqa: E402
from grounded_vision.perception.regions import extract_part_regions, pool_region_features  # noqa: E402
from grounded_vision.reasoning.tree_head import CertifiableTreeHead  # noqa: E402
from grounded_vision.explanations.pyxai_backend import PyXAIBackend  # noqa: E402


def pool_part_embeddings(cfg, split, provider):
    """Pools GT part-region embeddings from the feature cache. Returns
    (sample_ids, labels, dict sample_id -> {part -> [D]})"""
    dataset = build_dataset(cfg, split)
    from grounded_vision.data.cub import CUBSample, _DiskRecord

    def meta(i):
        rec = dataset.samples[i]
        if isinstance(rec, CUBSample):
            return rec.sample_id, rec.part_boxes, rec.part_visibilities, rec.label
        boxes, vis = dataset._part_boxes_from_points(rec.part_points)
        return rec.sample_id, boxes, vis, rec.label

    ids, labels, pooled = [], [], {}
    for i in range(len(dataset)):
        sid, boxes, vis, label = meta(i)
        if not provider.has_cache or sid not in provider.index:
            continue
        fmap = torch.from_numpy(np.asarray(provider.feature_maps[provider.index[sid]], dtype=np.float32)).unsqueeze(0)
        regions = extract_part_regions(boxes, vis)
        p = pool_region_features(fmap, regions)
        pooled[sid] = {r.part_name: p[r.id].numpy() for r in regions if r.id in p}
        ids.append(sid)
        labels.append(label)
    return ids, np.array(labels), pooled


def build_similarity_features(pooled_by_id, ids, protos, proto_names):
    X = np.zeros((len(ids), len(proto_names)), dtype=np.float32)
    for row, sid in enumerate(ids):
        emb = pooled_by_id[sid]
        for col, (part, attr_id, k) in enumerate(proto_names):
            if part in emb:
                X[row, col] = float(emb[part] @ protos[attr_id][k])
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    parser.add_argument("--n-instances", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    embed_dim = cfg["perception"]["embedding_dim"]

    # ---- Fitted prototypes only (exemplar-backed, not random init) ----------
    ckpt = torch.load(os.path.join(cache_dir, "perception_checkpoint.pt"),
                      map_location="cpu", weights_only=False)
    bank = ckpt["prototype_bank"]
    meta = ckpt.get("prototype_meta", {})
    proto_rows = []  # (part, attr_id, k, vec)
    for attr_id, part, attr_name, val in CANONICAL_BOUND_ATTRIBUTES:
        m = meta.get(str(attr_id), meta.get(attr_id))
        if not m or "exemplar_sample_id" not in m[0]:
            continue  # random-init prototype: not a meaningful similarity anchor
        key = str(attr_id) if str(attr_id) in bank else attr_id
        tensors = bank[key]
        for k in range(tensors.shape[0]):
            proto_rows.append((part, attr_id, k, tensors[k].numpy().astype(np.float32)))
    proto_names = [(p, a, k) for (p, a, k, _) in proto_rows]
    protos = {}
    for p, a, k, vec in proto_rows:
        protos.setdefault(a, {})[k] = vec
    attr_lookup = {a_id: (an, v) for a_id, pt, an, v in CANONICAL_BOUND_ATTRIBUTES}
    feature_names = [
        f"sim({part}:{attr_lookup[a_id][0]}={attr_lookup[a_id][1]}#{k})"
        for (part, a_id, k) in proto_names
    ]
    print(f"--> {len(proto_names)} prototype-similarity features "
          f"({len(set(a for _, a, _ in proto_names))} exemplar-backed attributes)")

    # ---- Pool part embeddings for both splits -------------------------------
    train_ids, ytr, pooled_tr = pool_part_embeddings(cfg, "train", FeatureProvider(cache_dir, "train"))
    test_ids, yte, pooled_te = pool_part_embeddings(cfg, "test", FeatureProvider(cache_dir, "test"))
    Xtr = build_similarity_features(pooled_tr, train_ids, protos, proto_names)
    Xte = build_similarity_features(pooled_te, test_ids, protos, proto_names)
    print(f"--> train {Xtr.shape}, test {Xte.shape}")

    out_dir = os.path.join(cache_dir, "prototype_tree")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    rows = rng.choice(len(Xte), size=min(args.n_instances, len(Xte)), replace=False)

    candidates = ([("ccp", {"ccp_alpha": a, "min_samples_leaf": 4})
                   for a in (0.00019, 0.00023, 0.00031, 0.00050)]
                  + [("leaves", {"max_leaf_nodes": nl, "min_samples_leaf": 4})
                     for nl in (32, 64, 128, 256, 512)])

    results = []
    for fam, params in candidates:
        dt = DecisionTreeClassifier(random_state=42, **params)
        dt.fit(Xtr, ytr)
        head = CertifiableTreeHead(feature_names=feature_names)
        head.clf, head.is_fitted = dt, True
        backend = PyXAIBackend(head)
        preds = dt.predict(Xte)
        reason_sizes, path_lens, con_flips = [], [], []
        instances = []
        for i in rows:
            x = Xte[i]
            pred = int(preds[i])
            exp = backend.explain_sufficient(x, pred, sample_id=int(i))
            literals = [c.condition_str for c in exp.conditions]
            pl = len(head.get_decision_path(x))
            reason_sizes.append(len(literals))
            path_lens.append(pl)
            con = backend.explain_contrastive(x, pred, sample_id=int(i))
            if con is not None:
                xc = x.copy()
                for c in con.conditions_to_alter:
                    # finite values guaranteed to fall on the contrast side of t
                    xc[c.feature_idx] = c.threshold + 1.0 if c.operator == ">" else c.threshold - 1.0
                con_flips.append(int(dt.predict(xc.reshape(1, -1))[0]) != pred)
            instances.append({"sample_id": int(test_ids[i]), "x": [float(v) for v in x],
                              "prediction": pred, "label": int(yte[i]),
                              "native_reason_literals": literals,
                              "native_reason_size": len(literals), "path_length": pl})
        entry = {
            "family": fam, "params": params, "n_leaves": int(dt.get_n_leaves()),
            "train_acc": float(np.mean(dt.predict(Xtr) == ytr)),
            "test_acc": float(np.mean(preds == yte)),
            "reason_mean": float(np.mean(reason_sizes)),
            "reason_max": int(np.max(reason_sizes)),
            "path_mean": float(np.mean(path_lens)),
            "frac_reason_lt_path": float(np.mean(np.array(reason_sizes) < np.array(path_lens))),
            "contrastive_flip_rate": float(np.mean(con_flips)) if con_flips else None,
        }
        results.append(entry)
        print(f"{fam}_{list(params.values())[0]:<9.5f} | leaves {entry['n_leaves']:4d} | "
              f"test {entry['test_acc']*100:5.2f}% | reason {entry['reason_mean']:5.2f} "
              f"(max {entry['reason_max']:2d}) | path {entry['path_mean']:5.2f} | "
              f"compr {entry['frac_reason_lt_path']*100:4.1f}%")

        name = f"{fam}_{list(params.values())[0]:.5f}".replace(".", "p")
        tree = dt.tree_
        nodes = []
        for nid in range(tree.node_count):
            if tree.feature[nid] == -2:
                nodes.append({"f": -1, "l": -1, "r": -1, "c": int(np.argmax(tree.value[nid][0])), "t": 0.0})
            else:
                nodes.append({"f": int(tree.feature[nid]), "l": int(tree.children_left[nid]),
                              "r": int(tree.children_right[nid]), "c": -1,
                              "t": float(tree.threshold[nid])})
        with open(os.path.join(out_dir, f"proto_{name}.json"), "w") as f:
            json.dump({"n_features": int(Xte.shape[1]), "numeric": True,
                       "feature_names": feature_names, "nodes": nodes,
                       "instances": instances}, f)

    with open(os.path.join(out_dir, "prototype_tree_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {os.path.join(out_dir, 'prototype_tree_summary.json')}")


if __name__ == "__main__":
    main()
