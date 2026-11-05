"""Fraction of verbalized absence clauses that sit over unobserved regions.

Motivation (external review): an absence clause like "no white belly color"
is only as good as the observation behind it — when the part region was not
observed, the sentence states an absence the pipeline never saw past. This
script measures, over the test set: of all negated conditions in the
sufficient explanations, how many resolve to facts without provenance
(no region / no reader probability). Run after the parts.txt underscore fix
so 'unobserved' reflects genuine CUB visibility, not the name-mismatch bug.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import yaml

from grounded_vision.pipeline import (
    active_vocabulary, build_backbone, build_dataset, load_perception,
    perceive_sample, FeatureProvider,
)
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.explanations.evidence import EvidenceResolver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cub_bound_full.yaml")
    ap.add_argument("--regions", default="gt", choices=["gt", "predicted"])
    ap.add_argument("--limit", type=int, default=0, help="0 = full test split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cache_dir = cfg["data"]["cache_dir"]
    mode = RepresentationMode(cfg["experiment"].get("representation", "bound"))
    attributes = active_vocabulary(cfg)
    compiler = FeatureCompiler(mode=mode, attributes=attributes)
    dataset = build_dataset(cfg, "test")
    backbone = build_backbone(cfg)
    part_detector, attr_reader, prototype_bank, pmeta = load_perception(
        cache_dir, cfg["perception"]["embedding_dim"], attributes=attributes)
    provider = FeatureProvider(cache_dir, "test", backbone=backbone)
    import pickle
    tree_head = pickle.load(open(os.path.join(cache_dir, "tree_head.pkl"), "rb"))["tree_head"]
    backend = PyXAIBackend(tree_head)

    n_samples = args.limit or len(dataset.samples)
    counts = dict(samples=0, conditions=0, negated=0, positive=0,
                  negated_unobserved=0, positive_unobserved=0)
    parts_unobserved: dict = {}
    t0 = time.time()
    for idx in range(n_samples):
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
        cards = EvidenceResolver(result.db).resolve_sufficient_explanation(suff)
        counts["samples"] += 1
        for c, card in zip(suff.conditions, cards):
            counts["conditions"] += 1
            observed = card.bbox is not None
            if c.operator == "<=":
                counts["negated"] += 1
                if not observed:
                    counts["negated_unobserved"] += 1
                    part = c.feature_name.split(":", 1)[0]
                    parts_unobserved[part] = parts_unobserved.get(part, 0) + 1
            else:
                counts["positive"] += 1
                if not observed:
                    counts["positive_unobserved"] += 1
        if (idx + 1) % 500 == 0:
            r = counts["negated_unobserved"] / max(counts["negated"], 1)
            print(f"[{idx+1}/{n_samples}] absence-over-unobserved {r:.1%} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    report = {
        "config": args.config,
        "regions": args.regions,
        "note": ("run after the parts.txt underscore fix; deployed tree_head "
                 "and perception checkpoint predate that fix"),
        **counts,
        "frac_absence_over_unobserved": counts["negated_unobserved"] / max(counts["negated"], 1),
        "frac_all_clauses_unobserved": (counts["negated_unobserved"] + counts["positive_unobserved"])
        / max(counts["conditions"], 1),
        "unobserved_absence_by_part": dict(sorted(parts_unobserved.items(),
                                                  key=lambda kv: -kv[1])),
    }
    out = args.out or os.path.join(
        cache_dir, f"unobserved_absence_{args.regions}_regions.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("unobserved_absence_by_part", "note")}, indent=2))
    print("by part:", report["unobserved_absence_by_part"])
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
