"""Reliability-ordered prime implicants + selective-curve confound controls.

PREREGISTERED POLICY (written before running): during backward elimination,
attempt to drop clauses in ASCENDING reliability order — least reliable
first. Reliability(c) = reader probability the clause is true of the image:
present clause -> p; absent clause -> 1 - p; no provenance -> 0.5 (neutral).
Rationale under the independence model P(all correct) ~= prod(p_i): dropping
the least reliable first (a) removes the worst factor whenever it is
droppable and (b) biases the surviving set toward high-reliability clauses.
Predicted effect: higher P(all correct) at full coverage, at some cost in
reason size. No calibration is fit; no test-side selection between orders.

Also answers the confound controls on the selective curve: AUROC of clause
count alone, within-strata AUROC, vacuous-pass handling for reasons with
zero present clauses, Wilson intervals at the tau=0.8 operating point, and
the present/absent composition of reasons.
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
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier
from grounded_vision.explanations.schema import (
    ExplanationCondition, VerificationStatus,
)
from grounded_vision.explanations.evidence import EvidenceResolver


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def reliability_of(cond, card):
    if card is None or card.confidence is None:
        return 0.5
    return card.confidence if cond.operator == ">" else 1.0 - card.confidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cub_bound_full_v2.yaml")
    ap.add_argument("--regions", default="gt")
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
    verifier = TreeSufficiencyVerifier(tree_head.clf, compiler.feature_names)

    train_ds = build_dataset(cfg, "train")
    cls_votes = {}
    for i in range(len(train_ds.samples)):
        _, _, _, labels = train_ds.meta(i)
        cls_votes.setdefault(train_ds.samples[i].label, []).append(labels)
    cls_majority = {c: (np.array(v).mean(axis=0) >= 0.5).astype(int)
                    for c, v in cls_votes.items()}

    name_to_col = {f"{p}:{a}={v}": j for j, (aid, p, a, v) in enumerate(attributes)}
    gt_cache = {}

    def clause_gt(sample, cond):
        name = cond.feature_name
        if name.startswith("has_part("):
            part = name[9:-1]
            v = 1 if sample.part_visibilities.get(part, 0) > 0.5 else 0
            return v, v
        if name not in name_to_col:
            return None, None
        if sample.sample_id not in gt_cache:
            gt_cache[sample.sample_id] = (np.array(sample.attribute_labels),
                                          cls_majority[sample.label])
        gt, gt_dn = gt_cache[sample.sample_id]
        j = name_to_col[name]
        return int(gt[j]), int(gt_dn[j])

    def reason_stats(sample, conds, cards_by_feat):
        ok, ok_dn, pres, pres_tp, absn, abs_tp = True, True, 0, 0, 0, 0
        for c in conds:
            g, g_dn = clause_gt(sample, c)
            if g is None:
                continue
            present = c.operator == ">"
            correct = (1 if present else 0) == g
            correct_dn = (1 if present else 0) == g_dn
            if present:
                pres += 1; pres_tp += int(correct)
            else:
                absn += 1; abs_tp += int(correct)
            ok &= correct
            ok_dn &= correct_dn
        return dict(all_correct=bool(ok), all_correct_dn=bool(ok_dn),
                    present=pres, present_tp=pres_tp, absent=absn, absent_tp=abs_tp)

    t0 = time.time()
    rows = []
    for idx in range(args.limit or len(dataset.samples)):
        sample = dataset[idx]
        res = perceive_sample(sample, provider, pd_, ar, pb, region_mode=args.regions,
                              probability_threshold=0.5, include_relations=False,
                              attributes=attributes)
        x = compiler.compile_database(res.db, use_probabilities=False)
        pred = int(tree_head.predict(x.reshape(1, -1))[0])
        default = backend.explain_sufficient(x, pred, sample_id=sample.sample_id)
        cards = EvidenceResolver(res.db).resolve_sufficient_explanation(default)
        cards_by_feat = {(c.feature_idx, c.operator): card
                         for c, card in zip(default.conditions, cards)}

        # ---- reliability-ordered backward elimination --------------------
        path = tree_head.get_decision_path(x)
        initial = [ExplanationCondition(
                       feature_idx=s.feature_idx, feature_name=s.feature_name,
                       operator="<=" if s.is_left else ">",
                       threshold=s.threshold, observed_value=s.value)
                   for s in path]
        rel = [reliability_of(c, cards_by_feat.get((c.feature_idx, c.operator)))
               for c in initial]
        minimal = list(initial)
        for cand in sorted(range(len(initial)), key=lambda i: rel[i]):
            trial = [c for c in minimal if c is not initial[cand]]
            if not trial and len(verifier.leaves) > 1:
                continue
            cert = verifier.verify_sufficiency(trial, pred)
            if cert.status == VerificationStatus.VERIFIED:
                minimal = trial
        reliab_ok = verifier.verify_sufficiency(minimal, pred).status == VerificationStatus.VERIFIED

        d_stats = reason_stats(sample, default.conditions, cards_by_feat)
        r_stats = reason_stats(sample, minimal, cards_by_feat)
        min_pres_conf = None
        for c, card in zip(default.conditions, cards):
            if c.operator == ">" and card.confidence is not None:
                v = card.confidence
                min_pres_conf = v if min_pres_conf is None else min(min_pres_conf, v)
        rows.append(dict(
            n_clauses=default.sparsity, n_present=d_stats["present"],
            min_present_conf=min_pres_conf,
            all_correct=d_stats["all_correct"], all_correct_dn=d_stats["all_correct_dn"],
            present_tp=d_stats["present_tp"], absent_tp=d_stats["absent_tp"],
            absent=d_stats["absent"],
            reliab_size=len(minimal), reliab_verified=bool(reliab_ok),
            reliab_all_correct=r_stats["all_correct"],
            reliab_all_correct_dn=r_stats["all_correct_dn"],
            reliab_present=r_stats["present"], reliab_present_tp=r_stats["present_tp"],
        ))
        if (idx + 1) % 1000 == 0:
            print(f"[{idx+1}] ({time.time()-t0:.0f}s)", flush=True)

    n = len(rows)
    allc = np.array([r["all_correct"] for r in rows])
    allc_dn = np.array([r["all_correct_dn"] for r in rows])
    n_pres = np.array([r["n_present"] for r in rows])
    n_cl = np.array([r["n_clauses"] for r in rows])
    conf = np.array([r["min_present_conf"] if r["min_present_conf"] is not None
                     else np.nan for r in rows])
    r_size = np.array([r["reliab_size"] for r in rows])
    r_allc = np.array([r["reliab_all_correct"] for r in rows])
    r_allc_dn = np.array([r["reliab_all_correct_dn"] for r in rows])

    # ---- confound controls ------------------------------------------------
    mask = ~np.isnan(conf)
    controls = {
        "n_reasons": n,
        "auroc_min_present_conf": auroc(conf[mask], allc[mask]),
        "auroc_n_present_alone": auroc(-n_pres.astype(float), allc),
        "auroc_n_clauses_alone": auroc(-n_cl.astype(float), allc),
        "zero_present_reasons": {
            "count": int((n_pres == 0).sum()),
            "all_correct_rate": float(allc[n_pres == 0].mean()) if (n_pres == 0).any() else None,
            "note": "min_present_conf is None -> excluded from every tau; no vacuous pass",
        },
        "auroc_within_n_present_strata": {},
    }
    for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 100)]:
        s = mask & (n_pres >= lo) & (n_pres <= hi)
        if s.sum() > 20:
            controls["auroc_within_n_present_strata"][f"{lo}-{min(hi,99)}+" if hi > 99 else f"{lo}"] = {
                "n": int(s.sum()), "auroc": auroc(conf[s], allc[s])}

    kept = mask & (conf >= 0.8)
    k = int(kept.sum())
    pres_tp_kept = sum(r["present_tp"] for i, r in enumerate(rows) if kept[i])
    pres_kept = sum(r["n_present"] for i, r in enumerate(rows) if kept[i])
    abs_tp_kept = sum(r["absent_tp"] for i, r in enumerate(rows) if kept[i])
    abs_kept = sum(r["absent"] for i, r in enumerate(rows) if kept[i])
    controls["tau_0.8_operating_point"] = {
        "n_kept": k,
        "p_all_correct": float(allc[kept].mean()),
        "p_all_correct_ci95_wilson": wilson(int(allc[kept].sum()), k),
        "present_precision": pres_tp_kept / max(pres_kept, 1),
        "present_precision_ci95_wilson": wilson(pres_tp_kept, pres_kept),
        "absent_clause_accuracy": abs_tp_kept / max(abs_kept, 1),
    }
    controls["clause_composition"] = {
        "mean_clauses": float(n_cl.mean()),
        "mean_present": float(n_pres.mean()),
        "mean_absent": float(np.mean([r["absent"] for r in rows])),
        "frac_clauses_absent": float(np.mean([r["absent"] / max(r["n_clauses"], 1) for r in rows])),
    }
    controls["reason_length_tax_per_clause_accuracy_for_p_all_correct_0.5"] = {
        f"k={k_}": round(0.5 ** (1.0 / k_), 4) for k_ in (2, 4, 6, 8, 10, 12, 15)}

    # ---- reliability-ordered vs default -----------------------------------
    comparison = {
        "default": {"mean_size": float(n_cl.mean()),
                    "p_all_correct": float(allc.mean()),
                    "p_all_correct_denoised": float(allc_dn.mean())},
        "reliability_ordered": {
            "verified_rate": float(np.mean([r["reliab_verified"] for r in rows])),
            "mean_size": float(r_size.mean()),
            "size_growth": float(r_size.mean() - n_cl.mean()),
            "p_all_correct": float(r_allc.mean()),
            "p_all_correct_denoised": float(r_allc_dn.mean()),
            "present_precision": (
                sum(r["reliab_present_tp"] for r in rows) /
                max(sum(r["reliab_present"] for r in rows), 1)),
        },
    }

    report = {
        "config": args.config, "regions": args.regions,
        "policy": "drop-least-reliable-first (preregistered; reliability p / 1-p / 0.5-no-prov)",
        "selective_confound_controls": controls,
        "reliability_ordered_minimization": comparison,
    }
    out = args.out or os.path.join(cache_dir, f"reliability_minimization_{args.regions}_regions.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
