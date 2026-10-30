"""
Script: analyze_full_run.py
===========================
Reviewer-driven analyses on a completed cub_bound_full run:

1. Concept-ceiling test: tree vs logistic regression vs MLP on the SAME
   predicted concept matrix, plus the same heads on GROUND-TRUTH concept
   labels (separates concept-set ceiling from perception error).
2. Explanation compression: |sufficient reason| / |decision path| per image,
   plus verified necessity (removing any single condition falsifies).
3. Per-attribute AUROC/F1 and cross-reference with the features the tree
   actually splits on.
4. Explanation-relative interventions: flipping in-reason concepts vs
   flipping certified contrastive conditions.
"""

from __future__ import annotations
import os
import json
import argparse
import pickle
import yaml
import numpy as np
from tqdm import tqdm

from grounded_vision.pipeline import build_dataset, load_perception, perceive_sample, FeatureProvider
from grounded_vision.data.cub import CANONICAL_BOUND_ATTRIBUTES
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier
from grounded_vision.explanations.schema import VerificationStatus


def compile_matrices(cfg, split, provider, part_detector, attr_reader, bank, region_mode, attributes=None):
    dataset = build_dataset(cfg, split)
    from grounded_vision.data.cub import CANONICAL_BOUND_ATTRIBUTES as _CANON
    attributes = attributes or _CANON
    compiler = FeatureCompiler(mode=RepresentationMode(cfg["experiment"]["representation"]), attributes=attributes)
    threshold = cfg["predicates"].get("probability_threshold", 0.5)
    dbs, labels, gt_rows = [], [], []
    n_missing = 0
    for i in tqdm(range(len(dataset)), desc=f"compile {split}"):
        sample = dataset[i]
        if not provider.has_cache or sample.sample_id not in provider.index:
            n_missing += 1
            continue
        result = perceive_sample(sample, provider, part_detector, attr_reader, bank,
                                 region_mode=region_mode, probability_threshold=threshold,
                                 attributes=attributes)
        dbs.append(result.db)
        labels.append(sample.label)
        # Ground-truth concept vector aligned to compiler.feature_names
        gt_vec = np.zeros(compiler.num_features, dtype=np.float32)
        attr_lookup = {(part, attr, val): idx for idx, (a_id, part, attr, val)
                       in enumerate(attributes)}
        parts_lookup = {f"has_part({part})": part for _, part, _, _ in attributes if part != "bird"}
        for fname, j in compiler.feature_to_idx.items():
            if fname in parts_lookup:
                gt_vec[j] = 1.0 if sample.part_visibilities.get(parts_lookup[fname], 0.0) >= threshold else 0.0
            elif ":" in fname:
                part, attr_val = fname.split(":", 1)
                attr, val = attr_val.split("=", 1)
                if (part, attr, val) in attr_lookup:
                    gt_vec[j] = sample.attribute_labels[attr_lookup[(part, attr, val)]]
        gt_rows.append(gt_vec)
    X = compiler.compile_batch(dbs, use_probabilities=False)
    return X, np.array(labels), np.array(gt_rows), compiler, n_missing


def ceiling_test(Xtr, ytr, Xte, yte, depth, leaf):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    heads = {}
    tree = CertifiableTreeHead(max_depth=depth, min_samples_leaf=leaf)
    tree.fit(Xtr, ytr)
    heads["tree_d%d" % depth] = tree
    heads["logreg_C1"] = LogisticRegression(max_iter=3000, C=1.0)
    heads["mlp_128x128"] = MLPClassifier(hidden_layer_sizes=(128, 128), early_stopping=True,
                                         validation_fraction=0.1, max_iter=500, random_state=42)
    out = {}
    for name, h in heads.items():
        h.fit(Xtr, ytr)
        out[name] = {
            "train_acc": float(np.mean(h.predict(Xtr) == ytr)),
            "test_acc": float(np.mean(h.predict(Xte) == yte)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub_bound_full.yaml")
    parser.add_argument("--regions", type=str, default="gt", choices=["gt", "predicted"])
    parser.add_argument("--necessity-sample", type=int, default=500)
    parser.add_argument("--flip-sample", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg["data"]["cache_dir"]
    embed_dim = cfg["perception"]["embedding_dim"]
    threshold = cfg["predicates"].get("probability_threshold", 0.5)

    from grounded_vision.pipeline import active_vocabulary
    attributes = active_vocabulary(cfg)
    part_detector, attr_reader, bank, _ = load_perception(cache_dir, embed_dim, attributes=attributes)

    print("==> Compiling concept matrices (predicted + ground truth)...")
    train_provider = FeatureProvider(cache_dir, "train")
    Xtr, ytr, GTtr, compiler, miss_tr = compile_matrices(
        cfg, "train", train_provider, part_detector, attr_reader, bank, args.regions, attributes)
    del train_provider
    test_provider = FeatureProvider(cache_dir, "test")
    Xte, yte, GTte, _, miss_te = compile_matrices(
        cfg, "test", test_provider, part_detector, attr_reader, bank, args.regions, attributes)

    # ---------- 1. Ceiling test ------------------------------------------------
    print("==> Ceiling test (same 36 concepts; heads differ)...")
    depth = cfg["reasoning"]["max_depth"]
    leaf = cfg["reasoning"]["min_samples_leaf"]
    ceiling = {
        "predicted_concepts": ceiling_test(Xtr, ytr, Xte, yte, depth, leaf),
        "gt_concepts": ceiling_test(GTtr, ytr, GTte, yte, depth, leaf),
        "concept_agreement_pred_vs_gt": {
            "train": float(np.mean(Xtr == GTtr)),
            "test": float(np.mean(Xte == GTte)),
        },
    }

    # ---------- 2. Compression + necessity on the deployed tree ----------------
    print("==> Explanation compression on deployed tree...")
    with open(os.path.join(cache_dir, "tree_head.pkl"), "rb") as f:
        tree_head = pickle.load(f)["tree_head"]
    backend = PyXAIBackend(tree_head)
    pred = tree_head.predict(Xte)
    sizes, path_lens, ratios = [], [], []
    contrast_found, contrast_sizes = 0, []
    for i in tqdm(range(len(Xte)), desc="explain"):
        exp = backend.explain_sufficient(Xte[i], int(pred[i]), sample_id=i)
        sizes.append(exp.sparsity)
        path_lens.append(len(tree_head.get_decision_path(Xte[i])))
        ratios.append(exp.sparsity / max(1, path_lens[-1]))
        con = backend.explain_contrastive(Xte[i], int(pred[i]), sample_id=i)
        if con is not None:
            contrast_found += 1
            contrast_sizes.append(len(con.conditions_to_alter))
    ratios = np.array(ratios)
    compression = {
        "n": len(Xte),
        "mean_reason_size": float(np.mean(sizes)),
        "mean_path_length": float(np.mean(path_lens)),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "p90_ratio": float(np.percentile(ratios, 90)),
        "frac_reason_equals_path": float(np.mean(np.array(sizes) == np.array(path_lens))),
        "frac_ratio_below_1": float(np.mean(ratios < 1.0)),
        "frac_ratio_below_half": float(np.mean(ratios < 0.5)),
        "contrastive_available_rate": contrast_found / len(Xte),
        "mean_contrastive_conditions": float(np.mean(contrast_sizes)) if contrast_sizes else None,
    }

    print(f"==> Necessity check on {args.necessity_sample} samples...")
    verifier = TreeSufficiencyVerifier(tree_head.clf, tree_head.feature_names)
    n_checks, n_necessary = 0, 0
    rng = np.random.RandomState(42)
    for i in rng.choice(len(Xte), size=min(args.necessity_sample, len(Xte)), replace=False):
        exp = backend.explain_sufficient(Xte[i], int(pred[i]), sample_id=int(i))
        for j in range(len(exp.conditions)):
            subset = [c for k, c in enumerate(exp.conditions) if k != j]
            n_checks += 1
            if verifier.verify_sufficiency(subset, int(pred[i])).status != VerificationStatus.VERIFIED:
                n_necessary += 1
    compression["necessity_rate"] = n_necessary / max(1, n_checks)
    compression["necessity_checks"] = n_checks

    # ---------- 3. Per-attribute quality vs tree usage --------------------------
    print("==> Per-attribute AUROC/F1 vs tree split usage...")
    from sklearn.metrics import roc_auc_score, f1_score
    tree = tree_head.clf.tree_
    usage = {}
    for f_idx in tree.feature:
        if f_idx >= 0:
            usage[f_idx] = usage.get(f_idx, 0) + 1
    attr_stats = []
    for f_idx, name in enumerate(tree_head.feature_names):
        if ":" not in name:
            continue
        part, attr_val = name.split(":", 1)
        attr, val = attr_val.split("=", 1)
        col = compiler.feature_to_idx[name]
        p, g = Xte[:, col], GTte[:, col]
        entry = {"feature": name, "positive_rate_gt": float(np.mean(GTte[:, col]))}
        if len(np.unique(g)) > 1 and len(np.unique(p)) > 1:
            entry["auroc"] = float(roc_auc_score(g, p))
            entry["f1"] = float(f1_score(g, p > 0.5, zero_division=0))
        else:
            entry["auroc"] = None
            entry["f1"] = None
        entry["tree_split_count"] = usage.get(f_idx, 0)
        attr_stats.append(entry)

    # ---------- 4. Explanation-relative interventions ---------------------------
    print(f"==> In-reason vs contrastive flips on {args.flip_sample} samples...")
    flip_idx = rng.choice(len(Xte), size=min(args.flip_sample, len(Xte)), replace=False)
    in_reason_flips, contrast_flips = [], []
    for i in flip_idx:
        exp = backend.explain_sufficient(Xte[i], int(pred[i]), sample_id=int(i))
        cols = [c.feature_idx for c in exp.conditions]
        Xi = Xte[i].copy()
        Xi[cols] = 1.0 - Xi[cols]
        in_reason_flips.append(int(tree_head.predict(Xi.reshape(1, -1))[0]) != int(pred[i]))
        con = backend.explain_contrastive(Xte[i], int(pred[i]), sample_id=int(i))
        if con is not None:
            cols_c = [c.feature_idx for c in con.conditions_to_alter]
            Xc = Xte[i].copy()
            Xc[cols_c] = 1.0 - Xc[cols_c]
            contrast_flips.append(int(tree_head.predict(Xc.reshape(1, -1))[0]) != int(pred[i]))
    interventions = {
        "flip_all_sufficient_reason_conditions_rate": float(np.mean(in_reason_flips)),
        "flip_contrastive_conditions_rate": float(np.mean(contrast_flips)) if contrast_flips else None,
        "n_evaluated": len(flip_idx),
    }

    report = {
        "config": args.config,
        "region_mode": args.regions,
        "skipped_from_cache": {"train": miss_tr, "test": miss_te},
        "ceiling_test": ceiling,
        "compression_and_minimality": compression,
        "attribute_stats": attr_stats,
        "explanation_relative_interventions": interventions,
    }
    out_path = os.path.join(cache_dir, "results", f"analysis_{args.regions}_regions.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("CEILING TEST (test acc / train acc)")
    for cond in ["predicted_concepts", "gt_concepts"]:
        print(f"  {cond}:")
        for head, m in ceiling[cond].items():
            print(f"    {head:14s} test {m['test_acc']*100:5.2f}%  train {m['train_acc']*100:5.2f}%")
    print(f"  concept agreement pred-vs-gt: test {ceiling['concept_agreement_pred_vs_gt']['test']*100:.1f}%")
    print("COMPRESSION")
    print(f"  reason size / path length: mean {compression['mean_ratio']:.3f} "
          f"median {compression['median_ratio']:.3f} | ==path {compression['frac_reason_equals_path']*100:.1f}% "
          f"| <0.5 {compression['frac_ratio_below_half']*100:.1f}%")
    print(f"  necessity rate (single-condition removal falsifies): {compression['necessity_rate']*100:.2f}% "
          f"over {n_checks} checks")
    print(f"  contrastive available: {compression['contrastive_available_rate']*100:.1f}% "
          f"(mean {compression['mean_contrastive_conditions']:.2f} conditions)")
    print("INTERVENTIONS (explanation-relative)")
    print(f"  flip all in-reason concepts: {interventions['flip_all_sufficient_reason_conditions_rate']*100:.1f}% change")
    print(f"  flip contrastive conditions: {interventions['flip_contrastive_conditions_rate']*100:.1f}% change")
    used = sorted([a for a in attr_stats if a["tree_split_count"] > 0],
                  key=lambda a: -a["tree_split_count"])[:10]
    print("TOP TREE SPLITS vs attribute quality:")
    for a in used:
        print(f"  {a['feature']:34s} splits={a['tree_split_count']:3d} "
              f"f1={a['f1'] if a['f1'] is None else round(a['f1'],3)} "
              f"auroc={a['auroc'] if a['auroc'] is None else round(a['auroc'],3)} "
              f"pos={a['positive_rate_gt']:.3f}")
    print("=" * 70)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
