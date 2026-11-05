"""Polarity-resolved correctness, label noise controls, and the selective curve.

Answers three review points on the concept-gap numbers:
1. Precision/recall per polarity + lift over a "predict-absent-always"
   baseline, because absent-clause accuracy is inflated by CUB's base rate.
   Reported against raw per-image labels AND class-majority-denoised labels
   (a CUB attribute counts as present for an image if the majority of its
   class's TRAIN images are annotated with it).
2. Independence check: P(all clauses correct) vs the product of per-clause
   accuracies — is reason-level correctness more than arithmetic?
3. Selective grounded explanations: does reader confidence predict which
   certified reasons are image-correct? tau grid over the minimum reader
   probability in a reason -> coverage(tau), P(all clauses correct | kept),
   present-clause precision among kept, and the AUROC of min-confidence
   for predicting all-correct (the "can we tell?" number).
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


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cub_bound_full_v2.yaml")
    ap.add_argument("--regions", default="gt", choices=["gt", "predicted"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cache_dir = cfg["data"]["cache_dir"]
    attributes = active_vocabulary(cfg)
    compiler = FeatureCompiler(mode=RepresentationMode("bound"), attributes=attributes)
    dataset = build_dataset(cfg, "test")
    backbone = build_backbone(cfg)
    pd_, ar, pb, _ = load_perception(cache_dir, cfg["perception"]["embedding_dim"],
                                     attributes=attributes)
    provider = FeatureProvider(cache_dir, "test", backbone=backbone)
    tree_head = pickle.load(open(os.path.join(cache_dir, "tree_head.pkl"), "rb"))["tree_head"]
    backend = PyXAIBackend(tree_head)

    # class-majority-denoised labels: per class, majority over TRAIN images
    train_ds = build_dataset(cfg, "train")
    cls_votes = {}
    for i in range(len(train_ds.samples)):
        sid, _, _, labels = train_ds.meta(i)
        cls_votes.setdefault(train_ds.samples[i].label, []).append(labels)
    cls_majority = {
        c: (np.array(v).mean(axis=0) >= 0.5).astype(int)
        for c, v in cls_votes.items()
    }

    n_samples = args.limit or len(dataset.samples)
    pol = dict(present_total=0, present_tp=0, absent_total=0, absent_tp=0,
               present_total_dn=0, present_tp_dn=0, absent_total_dn=0, absent_tp_dn=0,
               gt_pos_in_clauses=0)
    reasons = []
    t0 = time.time()
    for idx in range(n_samples):
        sample = dataset[idx]
        gt = np.array(sample.attribute_labels)
        gt_dn = cls_majority[sample.label]
        vis = sample.part_visibilities

        result = perceive_sample(sample, provider, pd_, ar, pb, region_mode=args.regions,
                                 probability_threshold=0.5, include_relations=False,
                                 attributes=attributes)
        x = compiler.compile_database(result.db, use_probabilities=False)
        pred = int(tree_head.predict(x.reshape(1, -1))[0])
        suff = backend.explain_sufficient(x, pred, sample_id=sample.sample_id)
        cards = EvidenceResolver(result.db).resolve_sufficient_explanation(suff)

        min_conf_present, min_conf_all = [], []
        all_ok, all_ok_dn, n_chk = True, True, 0
        present_tp = present_n = 0
        for c, card in zip(suff.conditions, cards):
            name = c.feature_name
            if name.startswith("has_part("):
                part = name[9:-1]
                gt_v = 1 if vis.get(part, 0) > 0.5 else 0
                gt_v_dn = gt_v
                j = None
            elif ":" in name and "=" in name:
                j = next(k for k, (aid, p, a, v) in enumerate(attributes)
                         if f"{p}:{a}={v}" == name)
                gt_v, gt_v_dn = int(gt[j]), int(gt_dn[j])
            else:
                continue
            present = c.operator == ">"
            ok = (1 if present else 0) == gt_v
            ok_dn = (1 if present else 0) == gt_v_dn
            if present:
                pol["present_total"] += 1
                pol["present_tp"] += int(ok)
                pol["present_total_dn"] += 1
                pol["present_tp_dn"] += int(ok_dn)
                present_n += 1
                present_tp += int(ok)
                if gt_v == 1:
                    pol["gt_pos_in_clauses"] += 1
                if card.confidence is not None:
                    min_conf_present.append(card.confidence)
                    min_conf_all.append(card.confidence)
            else:
                pol["absent_total"] += 1
                pol["absent_tp"] += int(ok)
                pol["absent_total_dn"] += 1
                pol["absent_tp_dn"] += int(ok_dn)
                if card.confidence is not None:
                    min_conf_all.append(1.0 - card.confidence)
            all_ok &= ok
            all_ok_dn &= ok_dn
            n_chk += 1

        reasons.append({
            "sample_id": int(sample.sample_id),
            "n_clauses": n_chk,
            "n_present": present_n,
            "present_tp": present_tp,
            "all_correct": bool(all_ok),
            "all_correct_denoised": bool(all_ok_dn),
            "min_present_conf": min(min_conf_present) if min_conf_present else None,
            "min_clause_conf": min(min_conf_all) if min_conf_all else None,
        })
        if (idx + 1) % 1000 == 0:
            print(f"[{idx+1}/{n_samples}] ({time.time()-t0:.0f}s)", flush=True)

    # ---- polarity metrics + baselines ------------------------------------
    base_present = pol["gt_pos_in_clauses"] / max(pol["present_total"], 1)
    P = lambda a, b: pol[a] / max(pol[b], 1)
    polarity = {
        "present_precision_raw": P("present_tp", "present_total"),
        "present_precision_denoised": P("present_tp_dn", "present_total_dn"),
        "present_recall_among_certified": P("present_tp", "gt_pos_in_clauses"),
        "absent_accuracy_raw": P("absent_tp", "absent_total"),
        "absent_accuracy_denoised": P("absent_tp_dn", "absent_total_dn"),
        "always_absent_baseline_on_absent_clauses": 1 - (pol["gt_pos_in_clauses"] / max(pol["present_total"] + pol["absent_total"], 1)),
        "base_rate_present_among_certified_clauses": base_present,
    }

    # ---- independence check + selective curve ----------------------------
    clause_acc = (pol["present_tp"] + pol["absent_tp"]) / max(
        pol["present_total"] + pol["absent_total"], 1)
    mean_n = float(np.mean([r["n_clauses"] for r in reasons]))
    indep = float(clause_acc ** mean_n)
    observed = float(np.mean([r["all_correct"] for r in reasons]))

    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    curve = []
    scores, labels_sel = [], []
    for tau in taus:
        kept = [r for r in reasons if r["min_present_conf"] is not None
                and r["min_present_conf"] >= tau]
        if not kept:
            curve.append({"tau": tau, "coverage": 0.0})
            continue
        curve.append({
            "tau": tau,
            "coverage": len(kept) / len(reasons),
            "p_all_correct_given_kept": float(np.mean([r["all_correct"] for r in kept])),
            "p_all_correct_denoised_given_kept": float(np.mean([r["all_correct_denoised"] for r in kept])),
            "present_precision_given_kept": (
                sum(r["present_tp"] for r in kept) /
                max(sum(r["n_present"] for r in kept), 1)),
        })
    for r in reasons:
        if r["min_present_conf"] is not None:
            scores.append(r["min_present_conf"])
            labels_sel.append(r["all_correct"])

    report = {
        "config": args.config,
        "regions": args.regions,
        "n_reasons": len(reasons),
        "polarity": polarity,
        "independence_check": {
            "clause_accuracy": clause_acc,
            "mean_clauses_per_reason": mean_n,
            "product_of_clause_accuracy": indep,
            "observed_p_all_correct": observed,
        },
        "selective_curve_min_present_conf": curve,
        "auroc_min_present_conf_predicts_all_correct": auroc(
            np.array(scores), np.array(labels_sel)),
        "n_with_reader_conf": len(scores),
    }
    out = args.out or os.path.join(cache_dir, f"selective_explanations_{args.regions}_regions.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    with open(out.replace(".json", "_reasons.jsonl"), "w") as f:
        for r in reasons:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
