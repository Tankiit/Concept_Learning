"""Clause-level concept correctness of certified reasons (the gap metric).

For every certified sufficient reason over the test split, compares each
clause's PREDICTED concept polarity against the image's ground-truth CUB
attribute label (or part visibility, for has_part clauses):

    clause correct  <=>  (predicted present and GT label = 1)
                     or  (predicted absent  and GT label = 0)

Reports per-clause correctness (overall, by polarity, by whether the clause's
region was observed) and P(all clauses correct | certified reason). This is
the measured gap between soundness-over-predicted-concepts (guaranteed by the
certificate) and truth-over-the-image (not guaranteed, now quantified).
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time

import numpy as np
import yaml

from grounded_vision.pipeline import (
    active_vocabulary, build_backbone, build_dataset, load_perception,
    perceive_sample, FeatureProvider,
)
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
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
    tree_head = pickle.load(open(os.path.join(cache_dir, "tree_head.pkl"), "rb"))["tree_head"]
    backend = PyXAIBackend(tree_head)

    n_samples = args.limit or len(dataset.samples)
    clauses = dict(total=0, correct=0,
                   present_total=0, present_correct=0,
                   absent_total=0, absent_correct=0,
                   observed_total=0, observed_correct=0,
                   unobserved_total=0, unobserved_correct=0)
    reasons_all_correct = 0
    reasons_all_clauses_observed_correct = 0
    reasons_with_observed_clauses = 0
    n_reasons = 0
    by_part: dict = {}
    t0 = time.time()

    for idx in range(n_samples):
        sample = dataset[idx]
        gt_labels = {}
        for j, (attr_id, part, attr, val) in enumerate(attributes):
            gt_labels[f"{part}:{attr}={val}"] = int(sample.attribute_labels[j])

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

        n_reasons += 1
        reason_correct = True
        reason_observed_correct = True
        has_observed = False
        for c, card in zip(suff.conditions, cards):
            name = c.feature_name
            if name.startswith("has_part("):
                part = name[9:-1]
                gt = 1 if sample.part_visibilities.get(part, 0) > 0.5 else 0
            elif name in gt_labels:
                part = name.split(":", 1)[0]
                gt = gt_labels[name]
            else:  # relational / unknown vocabulary: not GT-checkable here
                continue
            predicted_present = c.operator == ">"
            correct = (1 if predicted_present else 0) == gt
            observed = card.bbox is not None

            clauses["total"] += 1
            clauses["correct"] += int(correct)
            key = "present" if predicted_present else "absent"
            clauses[f"{key}_total"] += 1
            clauses[f"{key}_correct"] += int(correct)
            ok = "observed" if observed else "unobserved"
            clauses[f"{ok}_total"] += 1
            clauses[f"{ok}_correct"] += int(correct)
            slot = by_part.setdefault(part, [0, 0])
            slot[0] += int(correct)
            slot[1] += 1
            reason_correct &= correct
            if observed:
                has_observed = True
                reason_observed_correct &= correct

        reasons_all_correct += int(reason_correct)
        reasons_with_observed_clauses += int(has_observed)
        reasons_all_clauses_observed_correct += int(has_observed and reason_observed_correct)

        if (idx + 1) % 500 == 0:
            print(f"[{idx+1}/{n_samples}] clause acc "
                  f"{clauses['correct']/max(clauses['total'],1):.3f} | "
                  f"reason-level {reasons_all_correct/n_reasons:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    def rate(a, b):
        return clauses[a] / max(clauses[b], 1)

    report = {
        "config": args.config,
        "regions": args.regions,
        "n_reasons": n_reasons,
        "clauses": clauses,
        "clause_correctness": {
            "overall": rate("correct", "total"),
            "predicted_present": rate("present_correct", "present_total"),
            "predicted_absent": rate("absent_correct", "absent_total"),
            "observed_regions": rate("observed_correct", "observed_total"),
            "unobserved_regions": rate("unobserved_correct", "unobserved_total"),
        },
        "p_all_clauses_correct_given_reason": reasons_all_correct / max(n_reasons, 1),
        "p_all_observed_clauses_correct_given_reason":
            reasons_all_clauses_observed_correct / max(reasons_with_observed_clauses, 1),
        "clause_correctness_by_part": {
            p: {"correct": v[0], "total": v[1], "rate": v[0] / v[1]}
            for p, v in sorted(by_part.items(), key=lambda kv: kv[1][1], reverse=True)
        },
    }
    out = args.out or os.path.join(
        cache_dir, f"concept_correctness_{args.regions}_regions.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("clause_correctness_by_part",)}, indent=2))
    print("by part (worst first):",
          {p: round(v["rate"], 3) for p, v in sorted(
              report["clause_correctness_by_part"].items(),
              key=lambda kv: kv[1]["rate"])[:6]})
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
