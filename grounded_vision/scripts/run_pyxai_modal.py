"""Modal app: run PyXAI minimum-cardinality sufficient reasons for the
grounded_vision concept tree on Linux (where PyXAI's bundled d4 compiler runs).

CPU-only, minutes of compute. Usage:

    modal run scripts/run_pyxai_modal.py --data <pyxai_export_nN.json>
"""

import json
import time

import modal

app = modal.App("grounded-vision-pyxai")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # pyxai -> cv2 needs libGL headless
    .pip_install("pyxai==2.0.1", "numpy")
)


@app.function(image=image, cpu=1, memory=2048, timeout=15 * 60)
def run_comparison(payload: dict) -> dict:
    from pyxai import Builder, Explaining

    n_features = payload["n_features"]
    nodes = payload["nodes"]
    numeric = bool(payload.get("numeric", False))

    def to_builder(nid: int):
        node = nodes[nid]
        if node["f"] == -1:
            return int(node["c"])
        left = to_builder(node["l"])
        right = to_builder(node["r"])
        if numeric:
            # sklearn: value <= t goes left; PyXAI GE with children in sklearn
            # order reproduces the partition (verified 100% against sklearn).
            return Builder.DecisionNode(
                node["f"] + 1, operator="GE", threshold=float(node["t"]),
                left=left, right=right,
            )
        # sklearn on binary inputs: value <= 0.5 (== 0) goes left, == 1 goes right
        return Builder.DecisionNode(
            node["f"] + 1, operator="EQ", threshold=1, left=left, right=right
        )

    tree = Builder.DecisionTree(
        n_features,
        to_builder(0),
        feature_names=[f"f{i+1}" for i in range(n_features)],
        force_features_equal_to_binaries=not numeric,
    )
    kind = "numerical" if numeric else "binary"
    features_type = {kind: [f"f{i+1}" for i in range(n_features)]}

    def flatten_reason(reason):
        # n=1 returns a flat tuple of literals (or a single literal when the
        # reason has size 1); normalize to a list of ints. None = PyXAI gave up
        # (e.g. d4 timeout on very large trees).
        if reason is None:
            return None
        if isinstance(reason, int):
            return [reason]
        reason = list(reason)
        if reason and isinstance(reason[0], int):
            return reason
        return [int(l) for l in reason[0]]

    results = []
    diag = {}
    for k, inst in enumerate(payload["instances"]):
        rec = {"sample_id": inst["sample_id"], "native_size": inst["native_reason_size"]}
        try:
            t0 = time.time()
            conv = float if numeric else int  # int() would flatten [-1, 1] similarities
            explainer = Explaining.initialize(
                tree, instance=tuple(conv(v) for v in inst["x"]), features_type=features_type
            )
            rec["pyxai_prediction"] = int(explainer.target_prediction)
            rec["prediction_match"] = rec["pyxai_prediction"] == inst["prediction"]
            if k == 0:
                diag["binary_representation"] = [int(v) for v in explainer.binary_representation]

            minimum = flatten_reason(explainer.minimal_sufficient_reason())
            irreducible = flatten_reason(explainer.sufficient_reason())
            rec["minimum_size"] = len(minimum)
            rec["minimum_literals"] = [int(l) for l in minimum]
            rec["pyxai_irreducible_size"] = len(irreducible)

            if numeric:
                # PyXAI re-indexes binaries per (feature, threshold) condition on
                # numeric trees; the size comparison is still apples-to-apples.
                rec["native_feature_ids"] = None
                rec["minimum_subset_of_native"] = None
                rec["native_is_sufficient_per_pyxai"] = None
            else:
                # With force_features_equal_to_binaries, literal k == feature k.
                rec["minimum_feature_ids"] = sorted(abs(int(l)) for l in minimum)
                native_ids = sorted({abs(int(l)) for l in inst["native_reason_literals"]})
                rec["native_feature_ids"] = native_ids
                rec["minimum_subset_of_native"] = set(rec["minimum_feature_ids"]) <= set(native_ids)
                rec["native_is_sufficient_per_pyxai"] = bool(
                    explainer.is_sufficient_reason(inst["native_reason_literals"])
                )
            rec["elapsed_s"] = round(time.time() - t0, 3)
            rec["status"] = "ok"
        except Exception as exc:  # keep going; record the failure
            rec["status"] = f"error: {type(exc).__name__}: {exc}"
        results.append(rec)

    return {
        "diagnostics": diag,
        "n_instances": len(results),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "results": results,
    }


@app.local_entrypoint()
def main(data: str):
    with open(data) as f:
        payload = json.load(f)
    out = run_comparison.remote(payload)
    out_path = data.replace(".json", "_pyxai_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    ok = [r for r in out["results"] if r.get("status") == "ok"]
    print("diagnostics:", out.get("diagnostics"))
    print(f"ok: {out['n_ok']}/{out['n_instances']}")
    if ok:
        import statistics as st
        native = [r["native_size"] for r in ok]
        minimum = [r["minimum_size"] for r in ok]
        pred_match = sum(1 for r in ok if r.get("prediction_match"))
        subset = sum(1 for r in ok if r.get("minimum_subset_of_native"))
        print(f"prediction agreement local-vs-pyxai: {pred_match}/{len(ok)}")
        print(f"native (greedy irreducible) size: mean {st.mean(native):.2f} min {min(native)} max {max(native)}")
        print(f"PyXAI MINIMUM size:                mean {st.mean(minimum):.2f} min {min(minimum)} max {max(minimum)}")
        smaller = sum(1 for r in ok if r["minimum_size"] < r["native_size"])
        print(f"instances where PyXAI minimum < native: {smaller}/{len(ok)}")
        print(f"PyXAI minimum is a subset of the native reason: {subset}/{len(ok)}")
    errs = [r for r in out["results"] if r.get("status") != "ok"]
    for r in errs[:3]:
        print("ERR", r["sample_id"], r["status"])
    print(f"saved: {out_path}")
