"""
Script: explain.py
==================
Generates certified explanations for test samples:
- Minimal sufficient condition set (Prime Implicants)
- Visual evidence linking (part bounding boxes and exemplar prototypes)
- Faithful natural language verbalization
- Strict faithfulness audit report
"""

from __future__ import annotations
import os
import json
import argparse
import pickle
import yaml
import numpy as np

from grounded_vision.data.cub import CUBDataset, DEFAULT_SPECIES
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


def main():
    parser = argparse.ArgumentParser(description="Generate certified explanations.")
    parser.add_argument("--config", type=str, default="configs/cub_bound.yaml")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    model_path = os.path.join(cache_dir, "tree_head.pkl")

    mode_str = cfg["experiment"].get("representation", "bound")
    mode = RepresentationMode(mode_str)
    compiler = FeatureCompiler(mode=mode)

    dataset = CUBDataset(
        data_dir=cfg["data"]["data_dir"],
        split="test",
        image_size=cfg["data"]["image_size"],
        use_mock_if_missing=True,
        num_mock_samples=args.sample_idx + 5,
    )
    sample = dataset[args.sample_idx]

    backbone = FrozenDINOBackbone(
        model_name=cfg["perception"]["backbone"],
        embedding_dim=cfg["perception"]["embedding_dim"],
        patch_size=cfg["perception"]["patch_size"],
        image_size=cfg["data"]["image_size"],
    )

    embed_dim = cfg["perception"]["embedding_dim"]
    attr_reader = AttributeReader(embedding_dim=embed_dim)
    prototype_bank = AttributePrototypeBank(embedding_dim=embed_dim)

    # 1. Perception
    feats = extract_backbone_features(backbone, sample.image_tensor.unsqueeze(0))
    regions = extract_part_regions(sample.part_boxes, sample.part_visibilities)
    pooled = pool_region_features(feats["feature_map"], regions)
    attr_probs, proto_matches = attr_reader.predict_all(pooled, regions, prototype_bank)

    p_out = PerceptionOutput(
        sample_id=sample.sample_id,
        regions=regions,
        part_visibilities=sample.part_visibilities,
        attribute_probabilities=attr_probs,
        prototype_matches=proto_matches,
        feature_embeddings=pooled,
    )

    # 2. Grounding
    db = ground_perception_output(
        p_out,
        probability_threshold=cfg["predicates"].get("probability_threshold", 0.5),
        include_relations=(mode == RepresentationMode.RELATIONAL),
        species_name=sample.species_name,
    )

    # 3. Feature Compilation
    x = compiler.compile_database(db, use_probabilities=False)

    # 4. Model Loading or On-the-fly fit for demonstration
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
            tree_head = saved["tree_head"]
    else:
        # Fit quick mock tree
        X_mock = np.zeros((10, compiler.num_features), dtype=np.float32)
        X_mock[0] = x
        X_mock[1:] = np.random.binomial(1, 0.3, size=(9, compiler.num_features))
        y_mock = np.array([sample.label] + [i % 5 for i in range(9)])
        tree_head = CertifiableTreeHead(
            max_depth=5,
            feature_names=compiler.feature_names,
            class_names=DEFAULT_SPECIES,
        )
        tree_head.fit(X_mock, y_mock)

    pred_class = int(tree_head.predict(x.reshape(1, -1))[0])
    pred_species = (
        tree_head.class_names[pred_class]
        if pred_class < len(tree_head.class_names)
        else str(pred_class)
    )

    # 5. Certified Explanation via PyXAI / TreeVerifier
    backend = PyXAIBackend(tree_head)
    suff_exp = backend.explain_sufficient(x, pred_class, sample_id=sample.sample_id)
    contrast_exp = backend.explain_contrastive(x, pred_class, sample_id=sample.sample_id)

    # 6. Evidence Grounding
    resolver = EvidenceResolver(db, raw_image=sample.image)
    evidence_cards = resolver.resolve_sufficient_explanation(suff_exp)

    # 7. Natural Language Verbalization
    nl_text = verbalize_sufficient_explanation(suff_exp, evidence_cards, style="scientific")

    # 8. Faithfulness Audit
    auditor = ExplanationAuditor()
    report = auditor.audit(nl_text, suff_exp, db)

    # Display Results
    print("=" * 70)
    print(f"SAMPLE ID: {sample.sample_id} | Ground Truth: {sample.species_name}")
    print(f"PREDICTED CLASS: {pred_species}")
    print(f"VERIFICATION STATUS: {suff_exp.certificate.status.value.upper()}")
    print(f"EXPLANATION SPARSITY: {suff_exp.sparsity} minimal conditions")
    print("-" * 70)
    print("MINIMAL SUFFICIENT CONDITIONS:")
    for i, c in enumerate(suff_exp.conditions):
        print(f"  [{i+1}] {c.condition_str} (observed: {c.observed_value:.2f})")
    print("-" * 70)
    print("VERBALIZED NATURAL LANGUAGE EXPLANATION:")
    print(f"  \"{nl_text}\"")
    print("-" * 70)
    print(f"AUDIT VERDICT: {report.audit_summary}")
    print("=" * 70)


if __name__ == "__main__":
    main()
