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
    nodes = payload.get("nodes")
    numeric = bool(payload.get("numeric", False))
    forest_payload = payload.get("forest")

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

    model = None
    if forest_payload is not None:
        # Ensemble payload: one node-list per tree; majority-vote forest
        trees = []
        for t_nodes in forest_payload:
            nodes = t_nodes
            trees.append(Builder.DecisionTree(
                n_features, to_builder(0),
                feature_names=[f"f{i+1}" for i in range(n_features)],
                force_features_equal_to_binaries=True,
            ))
        model = Builder.RandomForest(trees, n_classes=int(payload["n_classes"]),
                                     feature_names=[f"f{i+1}" for i in range(n_features)])
        nodes = None
    tree = model if model is not None else Builder.DecisionTree(
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
    count_cfg = payload.get("count_sufficient_reasons") or {}
    count_n = int(count_cfg.get("n_instances", 0))
    count_tl = count_cfg.get("time_limit")
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

            # minimal_sufficient_reason routes through preferred_sufficient_reason
            # (MaxSAT), which in pyxai 2.0.1 crashes with TypeError when the tree
            # uses only a subset of the declared binary features (unmapped
            # literals). Degrade gracefully; plain sufficient_reason() is unaffected.
            try:
                minimum = flatten_reason(explainer.minimal_sufficient_reason())
            except (TypeError, IndexError):
                minimum = None
                rec["minimum_error"] = (
                    "pyxai-2.0.1 minimal_sufficient_reason crash "
                    "(preferred-reason path; unmapped literals on partially-"
                    "used binary feature set)")
            irreducible = flatten_reason(explainer.sufficient_reason())
            rec["minimum_size"] = len(minimum) if minimum is not None else None
            rec["minimum_literals"] = (
                [int(l) for l in minimum] if minimum is not None else None)
            rec["pyxai_irreducible_size"] = (
                len(irreducible) if irreducible is not None else None)

            # ---- E0 uniqueness: total number of sufficient reasons (trees only;
            # no exclusions; fresh explainer so exclusion state never leaks).
            # n_sufficient_reasons() is broken in pyxai 2.0.1 (IndexError inside
            # n_sufficient_reasons_per_attribute); fall back to enumerating all
            # reasons with a time limit.
            if k < count_n and forest_payload is None:
                try:
                    uniq_expl = Explaining.initialize(
                        tree, instance=tuple(conv(v) for v in inst["x"]),
                        features_type=features_type,
                    )
                    try:
                        uniq = uniq_expl.n_sufficient_reasons(time_limit=count_tl)
                        rec["n_sufficient_reasons"] = (
                            int(uniq) if isinstance(uniq, int) else str(uniq))
                        rec["n_sufficient_reasons_method"] = "model_count"
                    except (IndexError, TypeError):
                        try:
                            reasons_all = uniq_expl.sufficient_reason(
                                n=Explaining.ALL, time_limit=count_tl)
                            if reasons_all is None:
                                rec["n_sufficient_reasons"] = "timeout"
                            elif isinstance(reasons_all, tuple) and reasons_all and isinstance(reasons_all[0], int):
                                rec["n_sufficient_reasons"] = 1  # single reason, not wrapped
                            else:
                                rec["n_sufficient_reasons"] = len(reasons_all)
                            rec["n_sufficient_reasons_method"] = "enumeration"
                        except Exception as exc:
                            rec["n_sufficient_reasons"] = f"error: {type(exc).__name__}"
                except Exception as exc:
                    rec["n_sufficient_reasons"] = f"error: {type(exc).__name__}"

            if numeric:
                # PyXAI re-indexes binaries per (feature, threshold) condition on
                # numeric trees; the size comparison is still apples-to-apples.
                rec["native_feature_ids"] = None
                rec["minimum_subset_of_native"] = None
                rec["native_is_sufficient_per_pyxai"] = None
            else:
                # With force_features_equal_to_binaries, literal k == feature k.
                # Audited BEFORE any exclusions are set on this explainer.
                rec["minimum_feature_ids"] = (
                    sorted(abs(int(l)) for l in minimum)
                    if minimum is not None else None)
                native_ids = sorted({abs(int(l)) for l in inst["native_reason_literals"]})
                rec["native_feature_ids"] = native_ids
                rec["minimum_subset_of_native"] = (
                    set(rec["minimum_feature_ids"]) <= set(native_ids)
                    if minimum is not None else None)
                rec["native_is_sufficient_per_pyxai"] = bool(
                    explainer.is_sufficient_reason(
                        tuple(int(l) for l in inst["native_reason_literals"]))
                )

            # ---- per-instance excluded features (grounding-driven abstention):
            # reason must never use the excluded (untrusted) concepts. In pyxai
            # 2.0.1 the abstention signal is an EMPTY LIST ([]) from the n=1
            # excluded call, not the documented None; a genuine empty reason
            # (trivially sufficient) comes back as the empty TUPLE ().
            # Applied AFTER the baseline audits so those stay exclusion-free.
            excluded = inst.get("excluded_features") or []
            if excluded:
                explainer.set_excluded_features(
                    [f"f{i + 1}" for i in excluded])
                raw_excl = explainer.sufficient_reason(n=1)
                rec["excluded_n"] = len(excluded)
                abstained = raw_excl is None or (
                    isinstance(raw_excl, list) and len(raw_excl) == 0)
                if abstained:
                    rec["abstained"] = True
                    rec["excluded_size"] = None
                else:
                    excl_reason = flatten_reason(raw_excl)
                    rec["abstained"] = False
                    rec["excluded_size"] = len(excl_reason)
                    rec["excluded_literals"] = [int(l) for l in excl_reason]
                    if not numeric:
                        # literal k == feature k: the reason must not touch any
                        # excluded feature (verification, not trust)
                        used = {abs(int(l)) - 1 for l in excl_reason}
                        rec["excluded_leak"] = sorted(used & set(excluded))
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
        minimum = [r["minimum_size"] for r in ok if r.get("minimum_size") is not None]
        pred_match = sum(1 for r in ok if r.get("prediction_match"))
        subset = sum(1 for r in ok if r.get("minimum_subset_of_native"))
        print(f"prediction agreement local-vs-pyxai: {pred_match}/{len(ok)}")
        print(f"native (greedy irreducible) size: mean {st.mean(native):.2f} min {min(native)} max {max(native)}")
        if minimum:
            print(f"PyXAI MINIMUM size ({len(minimum)} computed):     mean {st.mean(minimum):.2f} min {min(minimum)} max {max(minimum)}")
        else:
            print("PyXAI MINIMUM size: none computed (pyxai 2.0.1 preferred-reason bug)")
        smaller = sum(1 for r in ok
                      if r.get("minimum_size") is not None
                      and r["minimum_size"] < r["native_size"])
        print(f"instances where PyXAI minimum < native: {smaller}/{len(ok)}")
        print(f"PyXAI minimum is a subset of the native reason: {subset}/{len(ok)}")
    excl = [r for r in ok if "abstained" in r]
    if excl:
        covered = [r for r in excl if not r["abstained"]]
        leaks = [r for r in covered if r.get("excluded_leak")]
        print(f"exclusion arm: covered (non-empty reason) {len(covered)}/{len(excl)}; "
              f"abstained {len(excl) - len(covered)}; excluded-feature leaks: {len(leaks)}")
        if covered:
            import statistics as st
            print(f"excluded-reason size: mean "
                  f"{st.mean(r['excluded_size'] for r in covered):.2f}")
    uniq = [r for r in ok if isinstance(r.get("n_sufficient_reasons"), int)]
    if uniq:
        ones = sum(1 for r in uniq if r["n_sufficient_reasons"] == 1)
        print(f"uniqueness (E0): n_sufficient_reasons == 1 for {ones}/{len(uniq)} "
              f"(others: {[r['n_sufficient_reasons'] for r in uniq if r['n_sufficient_reasons'] != 1][:10]})")
    errs = [r for r in out["results"] if r.get("status") != "ok"]
    for r in errs[:3]:
        print("ERR", r["sample_id"], r["status"])
    print(f"saved: {out_path}")
