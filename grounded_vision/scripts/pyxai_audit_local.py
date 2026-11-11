"""
Script: pyxai_audit_local.py
============================
Real-PyXAI cross-check for the CBM/CEM/Boolean audit.  Runs UNDER
ltn_pyxai_starter/.venv (pyxai 2.0.1; verified working on macOS).

Reads the payload written by run_cbm_cem_audit.py --pyxai-export
(binary-input sklearn CART trees exported as node lists + instances
with the native minimal sufficient reason) and, per instance:

    1. prediction agreement  PyXAI tree prediction == recorded prediction
  2. irreducible reason    explainer.sufficient_reason() (subset-minimal
                           per PyXAI's own irreducibility check)
  3. native certified      explainer.is_sufficient_reason(native_lits)
                           -- does real PyXAI agree the committed native
                           reason is sufficient?
  4. size agreement        |PyXAI irreducible| == |native reason|

NOTE minimal_sufficient_reason() is deliberately NOT called: pyxai
2.0.1's preferred-reason path crashes (TypeError) on Linux/Modal and
DEADLOCKS on macOS for trees that use only a subset of the declared
binary features (5th documented pyxai 2.0.1 defect; the other four are
in docs/wiki/REVIEW_VALIDATION_ROADMAP.md).  A SIGALRM guard bounds
each instance anyway.

Tree conversion: sklearn binary splits (value <= 0.5 goes left) become
PyXAI DecisionNode(f, EQ, 1) with sklearn child order -- the same
conversion verified 100% against sklearn predictions in
run_pyxai_modal.py.

Usage:
    ltn_pyxai_starter/.venv/bin/python scripts/pyxai_audit_local.py \
        --data results/pyxai_audit_export.json \
        --out results/pyxai_audit_local.json
"""

from __future__ import annotations
import argparse
import json
import signal
import time
from collections import defaultdict


def flatten_reason(reason):
    """Normalize PyXAI's varying n=1 return shapes to a list of ints
    (signed literals); None means PyXAI gave up / abstained."""
    if reason is None:
        return None
    if isinstance(reason, int):
        return [reason]
    reason = list(reason)
    if reason and isinstance(reason[0], int):
        return reason
    return [int(l) for l in reason[0]]


def to_builder(node):
    from pyxai import Builder
    if node["f"] == -1:
        return int(node["c"])
    # sklearn on binary inputs: value <= 0.5 (== 0) goes left, == 1 right
    return Builder.DecisionNode(
        node["f"] + 1, operator="EQ", threshold=1,
        left=to_builder(node["l"]), right=to_builder(node["r"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pyxai import Builder, Explaining

    payload = json.load(open(args.data))
    n_features = payload["n_features"]
    feat_names = [f"f{i+1}" for i in range(n_features)]

    by_block = defaultdict(list)
    for inst in payload["instances"]:
        by_block[inst["block"]].append(inst)

    results = []
    t_start = time.time()
    for block, insts in by_block.items():
        nodes = payload["trees"][block]
        tree = Builder.DecisionTree(
            n_features, to_builder(nodes),
            feature_names=feat_names,
            force_features_equal_to_binaries=True,
        )
        features_type = {"binary": feat_names}
        print(f"--> block {block}: {len(insts)} instances", flush=True)
        for k, inst in enumerate(insts):
            rec = {"sample_id": inst["sample_id"], "block": block,
                   "native_size": inst["native_reason_size"]}
            try:
                t0 = time.time()

                def on_alarm(signum, frame):
                    raise TimeoutError("per-instance 60s guard")
                signal.signal(signal.SIGALRM, on_alarm)
                signal.alarm(60)
                explainer = Explaining.initialize(
                    tree, instance=tuple(int(v) for v in inst["x"]),
                    features_type=features_type)
                rec["pyxai_prediction"] = int(explainer.target_prediction)
                rec["prediction_match"] = bool(
                    rec["pyxai_prediction"] == inst["prediction"])
                irreducible = flatten_reason(explainer.sufficient_reason())
                rec["pyxai_irreducible_size"] = (
                    len(irreducible) if irreducible is not None else None)
                rec["pyxai_irreducible_literals"] = irreducible
                native_ids = sorted({abs(int(l))
                                     for l in inst["native_reason_literals"]})
                rec["native_feature_ids"] = native_ids
                irr_ids = (sorted({abs(int(l)) for l in irreducible})
                           if irreducible is not None else None)
                rec["irreducible_feature_ids"] = irr_ids
                rec["irreducible_matches_native_set"] = (
                    irr_ids == native_ids if irr_ids is not None else None)
                rec["native_is_sufficient_per_pyxai"] = None
                try:
                    rec["native_is_sufficient_per_pyxai"] = bool(
                        explainer.is_sufficient_reason(
                            tuple(int(l) for l in inst["native_reason_literals"])))
                except AttributeError:
                    rec["native_audit_error"] = "PrimeImplicantCNF AttributeError"
                signal.alarm(0)
                rec["elapsed_s"] = round(time.time() - t0, 3)
                rec["status"] = "ok"
            except Exception as exc:
                signal.alarm(0)
                rec["status"] = f"error: {type(exc).__name__}: {exc}"
            results.append(rec)
            if (k + 1) % 25 == 0:
                print(f"    {k+1}/{len(insts)} done", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    summary = {
        "n_instances": len(results),
        "n_ok": len(ok),
        "prediction_match": sum(1 for r in ok if r.get("prediction_match")),
        "native_is_sufficient_per_pyxai": sum(
            1 for r in ok if r.get("native_is_sufficient_per_pyxai") is True),
        "native_is_sufficient_errors": sum(
            1 for r in ok if "native_audit_error" in r),
        "irreducible_matches_native_set": sum(
            1 for r in ok if r.get("irreducible_matches_native_set") is True),
        "irreducible_size_mean": (
            sum(r["pyxai_irreducible_size"] for r in ok
                if r.get("pyxai_irreducible_size") is not None) /
            max(1, sum(1 for r in ok
                       if r.get("pyxai_irreducible_size") is not None))
            if any(r.get("pyxai_irreducible_size") is not None for r in ok)
            else None),
        "elapsed_total_s": round(time.time() - t_start, 1),
        "by_block": {},
    }
    for block in by_block:
        bok = [r for r in ok if r["block"] == block]
        summary["by_block"][block] = {
            "n": len(bok),
            "pred_match": sum(1 for r in bok if r.get("prediction_match")),
            "native_suff": sum(1 for r in bok
                               if r.get("native_is_sufficient_per_pyxai") is True),
            "native_suff_errors": sum(1 for r in bok if "native_audit_error" in r),
            "irreducible_matches_native": sum(
                1 for r in bok
                if r.get("irreducible_matches_native_set") is True),
            "irreducible_size_mean": (
                sum(r["pyxai_irreducible_size"] for r in bok
                    if r.get("pyxai_irreducible_size") is not None) /
                max(1, sum(1 for r in bok
                           if r.get("pyxai_irreducible_size") is not None))
                if any(r.get("pyxai_irreducible_size") is not None for r in bok)
                else None),
            "native_size_mean": (
                sum(r["native_size"] for r in bok) / max(1, len(bok))),
        }
    out = {"summary": summary, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary["by_block"], indent=1))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
