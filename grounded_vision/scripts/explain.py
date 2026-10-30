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
import argparse
import pickle
import yaml
import numpy as np

from grounded_vision.pipeline import (
    active_vocabulary,
    build_backbone,
    build_dataset,
    load_perception,
    perceive_sample,
    FeatureProvider,
)
from grounded_vision.data.cub import DEFAULT_SPECIES
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
    parser.add_argument("--regions", type=str, default="gt", choices=["gt", "predicted"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["data"]["cache_dir"]
    model_path = os.path.join(cache_dir, "tree_head.pkl")

    mode_str = cfg["experiment"].get("representation", "bound")
    mode = RepresentationMode(mode_str)
    attributes = active_vocabulary(cfg)
    compiler = FeatureCompiler(mode=mode, attributes=attributes)
    threshold = cfg["predicates"].get("probability_threshold", 0.5)

    dataset = build_dataset(cfg, "test")
    sample = dataset[args.sample_idx]

    backbone = build_backbone(cfg)
    embed_dim = cfg["perception"]["embedding_dim"]
    part_detector, attr_reader, prototype_bank, pmeta = load_perception(cache_dir, embed_dim, attributes=attributes)
    if not pmeta["checkpoint_loaded"]:
        print("WARNING: explaining with untrained perception readers.")
    provider = FeatureProvider(cache_dir, "test", backbone=backbone)

    # 1-2. Perception + grounding (shared pipeline path)
    result = perceive_sample(
        sample, provider, part_detector, attr_reader, prototype_bank,
        region_mode=args.regions,
        probability_threshold=threshold,
        include_relations=(mode == RepresentationMode.RELATIONAL),
        attributes=attributes,
    )
    db = result.db

    # 3. Feature Compilation
    x = compiler.compile_database(db, use_probabilities=False)

    # 4. Load the trained tree head (mock-tree fallback only for smoke runs)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
            tree_head = saved["tree_head"]
    else:
        print(f"WARNING: {model_path} not found; fitting a throwaway mock tree for demonstration.")
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
    if contrast_exp is not None:
        print("CONTRASTIVE CONDITIONS (alter to move away from prediction):")
        for c in contrast_exp.conditions_to_alter:
            print(f"  - {c.condition_str}")
        print("-" * 70)
    print("VERBALIZED NATURAL LANGUAGE EXPLANATION:")
    print(f"  \"{nl_text}\"")
    print("-" * 70)
    print(f"AUDIT VERDICT: {report.audit_summary}")
    print("=" * 70)


if __name__ == "__main__":
    main()
