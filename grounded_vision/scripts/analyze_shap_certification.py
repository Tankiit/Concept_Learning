"""Do SHAP's top-k features constitute a certificate? (vs post-hoc baselines)

For each test sample: take the certified sufficient reason's size k, take the
k highest-|SHAP| features for the predicted class, pin them at their observed
values, and ask the independent tree verifier whether that set is sufficient
for the prediction. The certified reason passes by construction (sanity
check); the interesting number is how often SHAP's matched-size set FAILS
certification — the exact, checkable version of "a ranking is not a proof."
"""
from __future__ import annotations
import argparse
import json
import os
import pickle

import numpy as np
import yaml

from grounded_vision.pipeline import (
    active_vocabulary, build_backbone, build_dataset, load_perception,
    perceive_sample, FeatureProvider,
)
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier
from grounded_vision.explanations.schema import ExplanationCondition, VerificationStatus


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cub_bound_full.yaml")
    ap.add_argument("--regions", default="gt", choices=["gt", "predicted"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import shap

    cfg = yaml.safe_load(open(args.config))
    cache_dir = cfg["data"]["cache_dir"]
    mode = RepresentationMode(cfg["experiment"].get("representation", "bound"))
    attributes = active_vocabulary(cfg)
    compiler = FeatureCompiler(mode=mode, attributes=attributes)
    dataset = build_dataset(cfg, "test")
    backbone = build_backbone(cfg)
    part_detector, attr_reader, prototype_bank, _ = load_perception(
        cache_dir, cfg["perception"]["embedding_dim"], attributes=attributes)
    provider = FeatureProvider(cache_dir, "test", backbone=backbone)
    tree_head = pickle.load(open(os.path.join(cache_dir, "tree_head.pkl"), "rb"))["tree_head"]
    backend = PyXAIBackend(tree_head)
    verifier = TreeSufficiencyVerifier(tree_head.clf, compiler.feature_names)
    explainer = shap.TreeExplainer(tree_head.clf)

    n = min(args.n, len(dataset.samples))
    stats = dict(n=0, cert_sufficient=0, shap_sufficient=0,
                 sizes=[], overlaps=[], min_k=[], min_k_excess=[], min_k_found=0, shap_ok_flags=[])
    for idx in range(n):
        sample = dataset[idx]
        result = perceive_sample(
            sample, provider, part_detector, attr_reader, prototype_bank,
            region_mode=args.regions,
            probability_threshold=cfg["predicates"].get("probability_threshold", 0.5),
            include_relations=(mode == RepresentationMode.RELATIONAL),
            attributes=attributes,
        )
        x = compiler.compile_database(result.db, use_probabilities=False)
        pred = int(tree_head.predict(x.reshape(1, -1))[0])
        suff = backend.explain_sufficient(x, pred, sample_id=sample.sample_id)
        k = suff.sparsity
        if k == 0:
            continue

        sv = explainer.shap_values(x.reshape(1, -1))
        if isinstance(sv, list):            # older shap: list per class
            class_sv = sv[pred]
        elif sv.ndim == 3:                  # newer shap: (n, f, classes)
            class_sv = sv[:, :, pred]
        else:                                # binary/regression path
            pos = np.searchsorted(tree_head.clf.classes_, pred)
            class_sv = sv[pos] if isinstance(sv, np.ndarray) and sv.shape[0] == len(tree_head.clf.classes_) else sv
        class_sv = np.atleast_2d(np.asarray(class_sv))
        top_k = np.argsort(-np.abs(class_sv[0]))[:k]

        def pin(fidx):
            return ExplanationCondition(
                feature_idx=int(fidx),
                feature_name=compiler.feature_names[int(fidx)],
                operator=">" if x[fidx] > 0.5 else "<=",
                threshold=0.5,
                observed_value=float(x[fidx]),
            )

        cert_conds = [pin(c.feature_idx) for c in suff.conditions]
        shap_conds = [pin(f) for f in top_k]
        cert_ok = verifier.verify_sufficiency(cert_conds, pred).status == VerificationStatus.VERIFIED
        shap_ok = verifier.verify_sufficiency(shap_conds, pred).status == VerificationStatus.VERIFIED

        # smallest k at which SHAP's top-k becomes sufficient (vs certified k)
        order = np.argsort(-np.abs(class_sv[0]))
        min_k = None
        cap = min(len(order), 3 * k)
        for kk in range(1, cap + 1):
            conds = [pin(f) for f in order[:kk]]
            if verifier.verify_sufficiency(conds, pred).status == VerificationStatus.VERIFIED:
                min_k = kk
                break

        stats["n"] += 1
        stats["cert_sufficient"] += int(cert_ok)
        stats["shap_sufficient"] += int(shap_ok)
        stats["sizes"].append(k)
        stats["overlaps"].append(
            len({c.feature_idx for c in suff.conditions} & set(top_k)) / k)
        stats["shap_ok_flags"].append(bool(shap_ok))
        if min_k is not None:
            stats["min_k"].append(min_k)
            stats["min_k_excess"].append(min_k - k)
            stats["min_k_found"] += 1
        if (idx + 1) % 200 == 0:
            print(f"[{idx+1}/{n}] shap top-k sufficient "
                  f"{stats['shap_sufficient']/stats['n']:.3f}", flush=True)

    flags = np.array(stats["shap_ok_flags"])
    rng = np.random.RandomState(42)
    n_boot = 1000
    boot = [
        float(rng.choice(flags, size=len(flags), replace=True).mean())
        for _ in range(n_boot)
    ]

    min_k = np.array(stats["min_k"]) if stats["min_k"] else np.array([np.nan])
    sizes = np.array(stats["sizes"])
    report = {
        "config": args.config,
        "regions": args.regions,
        "n": stats["n"],
        "mean_reason_size": float(np.mean(stats["sizes"])),
        "certified_reason_sufficient_rate": stats["cert_sufficient"] / stats["n"],
        "shap_topk_sufficient_rate": stats["shap_sufficient"] / stats["n"],
        "shap_topk_certification_failure_rate": 1 - stats["shap_sufficient"] / stats["n"],
        "shap_topk_failure_rate_ci95": [
            1 - float(np.percentile(boot, 97.5)),
            1 - float(np.percentile(boot, 2.5)),
        ],
        "mean_topk_overlap_with_certified_reason": float(np.mean(stats["overlaps"])),
        "shap_min_sufficient_k": {
            "found_within_3x_certified": stats["min_k_found"] / stats["n"],
            "mean": float(np.nanmean(min_k)),
            "median": float(np.nanmedian(min_k)),
            "mean_excess_over_certified": float(np.mean(stats["min_k_excess"])) if stats["min_k_excess"] else None,
        },
    }
    out = args.out or os.path.join(
        cache_dir, f"shap_certification_{args.regions}_regions.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
