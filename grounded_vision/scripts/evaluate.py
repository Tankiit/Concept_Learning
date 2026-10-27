"""
Script: evaluate.py
===================
Runs comprehensive evaluation across:
1. Perception: Part localization IoU and attribute classification F1
2. Downstream Classification: Accuracy, Macro-F1, Calibration (ECE)
3. Formal Explanations: Mathematical sufficiency rate, sparsity, faithfulness audit
4. Concept Interventions: Prediction flip rates under counterfactual interventions
"""

from __future__ import annotations
import os
import json
import argparse
import pickle
import yaml
import numpy as np
from tqdm import tqdm

from grounded_vision.data.cub import CUBDataset, DEFAULT_SPECIES, CANONICAL_BOUND_ATTRIBUTES
from grounded_vision.perception.backbone import FrozenDINOBackbone, extract_backbone_features
from grounded_vision.perception.regions import extract_part_regions, pool_region_features
from grounded_vision.perception.readers import PartDetector, AttributeReader, PerceptionOutput
from grounded_vision.perception.prototypes import AttributePrototypeBank
from grounded_vision.predicates.grounding import ground_perception_output
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
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
    parser.add_argument("--num-samples", type=int, default=40)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    results_dir = os.path.join(cache_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    mode_str = cfg["experiment"].get("representation", "bound")
    mode = RepresentationMode(mode_str)
    compiler = FeatureCompiler(mode=mode)

    print(f"--> Loading evaluation dataset ({args.num_samples} samples)...")
    dataset = CUBDataset(
        data_dir=cfg["data"]["data_dir"],
        split="test",
        image_size=cfg["data"]["image_size"],
        use_mock_if_missing=True,
        num_mock_samples=args.num_samples,
    )

    backbone = FrozenDINOBackbone(
        model_name=cfg["perception"]["backbone"],
        embedding_dim=cfg["perception"]["embedding_dim"],
        patch_size=cfg["perception"]["patch_size"],
        image_size=cfg["data"]["image_size"],
    )

    embed_dim = cfg["perception"]["embedding_dim"]
    part_detector = PartDetector(embedding_dim=embed_dim)
    attr_reader = AttributeReader(embedding_dim=embed_dim)
    prototype_bank = AttributePrototypeBank(embedding_dim=embed_dim)

    # 1. Evaluate Perception
    print("--> Evaluating Perception...")
    pred_boxes_list = []
    gt_boxes_list = []
    pred_attrs = []
    gt_attrs = []
    dbs = []
    labels = []

    for i in tqdm(range(len(dataset)), desc="Perception inference"):
        sample = dataset[i]
        feats = extract_backbone_features(backbone, sample.image_tensor.unsqueeze(0))
        cls_token = feats["cls_token"]
        fmap = feats["feature_map"]

        vis_preds, box_preds = part_detector(cls_token)
        pred_boxes_list.append(box_preds)
        gt_boxes_list.append(sample.part_boxes)

        regions = extract_part_regions(sample.part_boxes, sample.part_visibilities)
        pooled = pool_region_features(fmap, regions)

        attr_probs, proto_matches = attr_reader.predict_all(pooled, regions, prototype_bank)

        # Attribute probability vector
        p_vec = np.zeros(len(CANONICAL_BOUND_ATTRIBUTES))
        for a_id in range(len(CANONICAL_BOUND_ATTRIBUTES)):
            p_vec[a_id] = attr_probs.get(a_id, 0.0)
        pred_attrs.append(p_vec)
        gt_attrs.append(sample.attribute_labels)

        p_out = PerceptionOutput(
            sample_id=sample.sample_id,
            regions=regions,
            part_visibilities=sample.part_visibilities,
            attribute_probabilities=attr_probs,
            prototype_matches=proto_matches,
            feature_embeddings=pooled,
        )

        db = ground_perception_output(
            p_out,
            probability_threshold=cfg["predicates"].get("probability_threshold", 0.5),
            include_relations=(mode == RepresentationMode.RELATIONAL),
            species_name=sample.species_name,
        )
        dbs.append(db)
        labels.append(sample.label)

    perception_metrics = evaluate_perception_performance(
        pred_boxes_list,
        gt_boxes_list,
        np.array(pred_attrs),
        np.array(gt_attrs),
    )

    # 2. Evaluate Downstream Classifier
    print("--> Compiling Features & Evaluating Classifier...")
    X = compiler.compile_batch(dbs, use_probabilities=False)
    y = np.array(labels)

    # Fit or load tree
    tree_head = CertifiableTreeHead(
        max_depth=6,
        feature_names=compiler.feature_names,
        class_names=DEFAULT_SPECIES,
    )
    tree_head.fit(X, y)

    y_pred = tree_head.predict(X)
    y_probs = tree_head.predict_proba(X)
    cls_metrics = evaluate_classifier_performance(y, y_pred, y_probs)

    # 3. Evaluate Formal Explanations
    print("--> Generating & Auditing Formal Explanations...")
    backend = PyXAIBackend(tree_head)
    auditor = ExplanationAuditor()
    explanations = []
    audit_reports = []

    for i in tqdm(range(len(dataset)), desc="Explaining"):
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
    print(f"Perception Mean Part IoU:      {perception_metrics['mean_part_iou']:.4f}")
    print(f"Perception Attribute Acc:      {perception_metrics['attribute_accuracy'] * 100:.2f}%")
    print(f"Downstream Species Accuracy:   {cls_metrics['accuracy'] * 100:.2f}%")
    print(f"Downstream Macro-F1:           {cls_metrics['macro_f1']:.4f}")
    print(f"Explanation Sufficiency Rate:  {exp_metrics['sufficiency_rate'] * 100:.2f}%")
    print(f"Explanation Mean Sparsity:     {exp_metrics['mean_sparsity_conditions']:.2f} conditions")
    print(f"NL Faithfulness Rate:          {exp_metrics['nl_faithful_rate'] * 100:.2f}%")
    print(f"Mean Prediction Flip Rate:     {intervention_metrics['mean_prediction_flip_rate'] * 100:.2f}%")
    print("=" * 65)
    print(f"Results successfully saved to {report_path}")


if __name__ == "__main__":
    main()
