"""
Script: evaluate.py
===================
Runs comprehensive evaluation across:
1. Perception: Part localization IoU and attribute classification F1
2. Downstream Classification: Accuracy, Macro-F1, Calibration (ECE)
3. Formal Explanations: Mathematical sufficiency rate, sparsity, faithfulness audit
4. Concept Interventions: Prediction flip rates under counterfactual interventions

The tree head is LOADED from {cache_dir}/tree_head.pkl (trained by
train_head.py) so test metrics reflect honest train/test separation.
"""

from __future__ import annotations
import os
import json
import argparse
import pickle
import yaml
import numpy as np
from tqdm import tqdm

from grounded_vision.pipeline import (
    active_vocabulary,
    build_backbone,
    build_dataset,
    load_perception,
    perceive_sample,
    FeatureProvider,
)
from grounded_vision.data.cub import CANONICAL_BOUND_ATTRIBUTES
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.explanations.evidence import EvidenceResolver
from grounded_vision.language.templates import verbalize_sufficient_explanation
from grounded_vision.language.checks import ExplanationAuditor
from grounded_vision.evaluation.perception import evaluate_perception_performance
from grounded_vision.evaluation.classification import evaluate_classifier_performance
from grounded_vision.evaluation.explanations import evaluate_explanation_suite
from grounded_vision.evaluation.interventions import evaluate_interventions


def main():
    parser = argparse.ArgumentParser(description="Run complete grounded_vision benchmark.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--num-samples", type=int, default=0, help="Test samples to evaluate (0 = all)")
    parser.add_argument("--regions", type=str, default="gt", choices=["gt", "predicted"])
    parser.add_argument("--skip-explanations", action="store_true",
                        help="Skip per-sample explanation/audit loop for a faster perception+classification run")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    results_dir = os.path.join(cache_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    mode_str = cfg["experiment"].get("representation", "bound")
    mode = RepresentationMode(mode_str)
    attributes = active_vocabulary(cfg)
    compiler = FeatureCompiler(mode=mode, attributes=attributes)
    threshold = cfg["predicates"].get("probability_threshold", 0.5)

    model_path = os.path.join(cache_dir, "tree_head.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Train it first: python scripts/train_head.py --config {args.config}"
        )
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
        tree_head = saved["tree_head"]
    print(f"--> Loaded trained tree head from {model_path}")

    dataset = build_dataset(cfg, "test")
    backbone = build_backbone(cfg)
    embed_dim = cfg["perception"]["embedding_dim"]
    part_detector, attr_reader, prototype_bank, pmeta = load_perception(cache_dir, embed_dim, attributes=attributes)
    if not pmeta["checkpoint_loaded"]:
        print("WARNING: evaluating with untrained perception readers.")
    provider = FeatureProvider(cache_dir, "test", backbone=backbone)

    n = len(dataset) if args.num_samples <= 0 else min(len(dataset), args.num_samples)

    # 1. Evaluate Perception + compile grounded databases
    print(f"--> Perception inference on {n} test samples (regions={args.regions})...")
    pred_boxes_list = []
    gt_boxes_list = []
    pred_attrs = []
    gt_attrs = []
    dbs = []
    labels = []
    n_missing = 0

    for i in tqdm(range(n)):
        sample = dataset[i]
        if not provider.has_cache or sample.sample_id not in provider.index:
            n_missing += 1
            continue
        result = perceive_sample(
            sample, provider, part_detector, attr_reader, prototype_bank,
            region_mode=args.regions,
            probability_threshold=threshold,
            include_relations=(mode == RepresentationMode.RELATIONAL),
            attributes=attributes,
        )

        pred_boxes_list.append(result.box_preds)
        gt_boxes_list.append(sample.part_boxes)

        p_vec = np.zeros(len(attributes))
        for a_id in range(len(attributes)):
            p_vec[a_id] = result.output.attribute_probabilities.get(a_id, 0.0)
        pred_attrs.append(p_vec)
        gt_attrs.append(sample.attribute_labels)

        dbs.append(result.db)
        labels.append(sample.label)

    if n_missing:
        print(f"WARNING: {n_missing} test samples missing from feature cache were skipped.")
    if not dbs:
        raise RuntimeError("No cached test samples available; run cache_features.py --split test.")

    perception_metrics = evaluate_perception_performance(
        pred_boxes_list,
        gt_boxes_list,
        np.array(pred_attrs),
        np.array(gt_attrs),
    )

    # 2. Evaluate Downstream Classifier (tree trained on the train split)
    print("--> Compiling Features & Evaluating Classifier...")
    X = compiler.compile_batch(dbs, use_probabilities=False)
    y = np.array(labels)

    y_pred = tree_head.predict(X)
    y_probs = tree_head.predict_proba(X)
    cls_metrics = evaluate_classifier_performance(y, y_pred, y_probs)

    # 3. Evaluate Formal Explanations
    if args.skip_explanations:
        exp_metrics = {"skipped": True}
        audit_reports = []
    else:
        print("--> Generating & Auditing Formal Explanations...")
        backend = PyXAIBackend(tree_head)
        auditor = ExplanationAuditor()
        explanations = []
        audit_reports = []

        for i in tqdm(range(len(dbs)), desc="Explaining"):
            xi = X[i]
            pred_c = int(y_pred[i])
            db_i = dbs[i]
            sample_i = dataset[i]

            suff_exp = backend.explain_sufficient(xi, pred_c, sample_id=sample_i.sample_id)
            explanations.append(suff_exp)

            resolver = EvidenceResolver(db_i, raw_image=sample_i.image)
            ev_cards = resolver.resolve_sufficient_explanation(suff_exp)
            nl_text = verbalize_sufficient_explanation(suff_exp, ev_cards, style="scientific")
            report = auditor.audit(nl_text, suff_exp, db_i)
            audit_reports.append(report)

        exp_metrics = evaluate_explanation_suite(explanations, audit_reports)

    # 4. Evaluate Interventions
    print("--> Evaluating Concept Interventions...")
    intervention_metrics = evaluate_interventions(tree_head, compiler, X, y)

    # Combine all results
    full_report = {
        "experiment_name": cfg["experiment"]["name"],
        "representation": mode_str,
        "region_mode": args.regions,
        "n_test_samples": len(dbs),
        "perception": perception_metrics,
        "classification": cls_metrics,
        "explanations": exp_metrics,
        "interventions": intervention_metrics,
    }

    report_path = os.path.join(results_dir, "benchmark_results.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 65)
    print("           GROUNDED VISION BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Test samples:                 {len(dbs)}")
    print(f"Perception Mean Part IoU:     {perception_metrics['mean_part_iou']:.4f}")
    print(f"Perception Attribute Acc:     {perception_metrics['attribute_accuracy'] * 100:.2f}%")
    print(f"Downstream Species Accuracy:  {cls_metrics['accuracy'] * 100:.2f}%")
    print(f"Downstream Macro-F1:          {cls_metrics['macro_f1']:.4f}")
    if not args.skip_explanations:
        print(f"Explanation Sufficiency Rate: {exp_metrics['sufficiency_rate'] * 100:.2f}%")
        print(f"Explanation Mean Sparsity:    {exp_metrics['mean_sparsity_conditions']:.2f} conditions")
        print(f"NL Faithfulness Rate:         {exp_metrics['nl_faithful_rate'] * 100:.2f}%")
    print(f"Mean Prediction Flip Rate:    {intervention_metrics['mean_prediction_flip_rate'] * 100:.2f}%")
    print("=" * 65)
    print(f"Results successfully saved to {report_path}")


if __name__ == "__main__":
    main()
