"""
explain.py — PyXAI reasons + independent verifier (PyXAI env).
================================================================
Stage 4 (ltn_pyxai_starter/.venv).  Real PyXAI 2.0.1 reasons for the
distilled student tree, each cross-checked by a SECOND implementation
(utils/tree_ops.py: numpy routing + exact reachable-leaf DFS +
backward-elimination minimal reason).  Mutual audits both directions.

Per test instance (bank bits), writes results/reasons_K{K}.jsonl:
    sample_id, pred_tree (numpy router), pred_pyxai, prediction_match,
    pyxai_reason (signed ints, +k = feature k true), pyxai_size,
    native_reason, native_size, native_verified (DFS certificate),
    pyxai_is_sufficient_native (engine audits OUR reason),
    native_is_sufficient_pyxai (verifier audits PyXAI's reason),
    sets_match, sizes_match

Known pyxai 2.0.1 defects honoured: minimal_sufficient_reason is NEVER
called (crashes on Linux, deadlocks on arm64 macOS for trees that use
a subset of declared features).  SIGALRM guards each instance anyway.

Usage
-----
    python src/explain.py --classes 4
"""

from __future__ import annotations
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.tree_ops import TreeOps  # noqa: E402

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
RESULTS_DIR = PROJECT_ROOT / "results"


def flatten_reason(reason):
    if reason is None:
        return None
    if isinstance(reason, int):
        return [reason]
    reason = list(reason)
    if reason and isinstance(reason[0], int):
        return reason
    return [int(l) for l in reason[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--tree", default=None)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--native-only", action="store_true",
                    help="skip PyXAI; write native verifier reasons only "
                         "(for pyxai-less environments / tests)")
    args = ap.parse_args()
    tree_path = args.tree or str(
        Path(args.bank) / f"tree_K{args.classes}.json")
    out = args.out or str(RESULTS_DIR / f"reasons_K{args.classes}.jsonl")

    tree_json = json.loads(Path(tree_path).read_text())
    ops = TreeOps(tree_json)
    te = np.load(Path(args.bank) / "bank_test.npz")
    mask = te["labels"] < args.classes
    ids = te["sample_ids"][mask]
    X = te["bits"][mask].astype(np.int64)
    if args.limit:
        ids, X = ids[: args.limit], X[: args.limit]

    ptree = None
    features_type = None
    if not args.native_only:
        from pyxai import Builder, Explaining

        def to_builder(node):
            if "c" in node and "f" not in node:
                return int(node["c"])
            return Builder.DecisionNode(
                node["f"] + 1, operator="EQ", threshold=1,
                left=to_builder(node["l"]), right=to_builder(node["r"]))

        feat_names = [f"f{i+1}" for i in range(ops.n_features)]
        ptree = Builder.DecisionTree(
            ops.n_features, to_builder(tree_json["root"]),
            feature_names=feat_names,
            force_features_equal_to_binaries=True)
        features_type = {"binary": feat_names}

    t0 = time.time()
    n_rows = 0
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        for i in range(len(ids)):
            rec = {"sample_id": int(ids[i]), "K": int(args.classes)}
            try:
                def on_alarm(signum, frame):
                    raise TimeoutError("60s per-instance guard")
                signal.signal(signal.SIGALRM, on_alarm)
                signal.alarm(60)
                x = X[i]
                native = ops.minimal_reason(x, ops.predict_one(x))
                rec["pred_tree"] = int(ops.predict_one(x))
                rec["native_reason"] = [(int(f) + 1, int(b)) for f, b in native]
                rec["native_size"] = len(native)
                rec["native_verified"] = bool(
                    ops.sufficient(native, rec["pred_tree"]))
                if args.native_only:
                    rec["status"] = "ok"
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_rows += 1
                    continue
                explainer = Explaining.initialize(
                    ptree, instance=tuple(int(v) for v in x),
                    features_type=features_type)
                rec["pred_pyxai"] = int(explainer.target_prediction)
                rec["prediction_match"] = bool(
                    rec["pred_pyxai"] == rec["pred_tree"])
                pyx = flatten_reason(explainer.sufficient_reason())
                rec["pyxai_reason"] = pyx
                rec["pyxai_size"] = len(pyx) if pyx is not None else None
                signed_native = [(f + 1) if b else -(f + 1) for f, b in native]
                rec["pyxai_is_sufficient_native"] = bool(
                    explainer.is_sufficient_reason(tuple(signed_native)))
                if pyx is not None:
                    rec["native_is_sufficient_pyxai"] = bool(
                        ops.sufficient([(abs(int(l)) - 1,
                                         1 if int(l) > 0 else 0) for l in pyx],
                                       rec["pred_tree"]))
                    rec["sets_match"] = bool(
                        sorted(abs(int(l)) for l in pyx)
                        == sorted(f + 1 for f, _ in native))
                    rec["sizes_match"] = bool(
                        rec["pyxai_size"] == rec["native_size"])
                signal.alarm(0)
                rec["status"] = "ok"
            except Exception as exc:
                signal.alarm(0)
                rec["status"] = f"error: {type(exc).__name__}: {exc}"
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_rows += 1
    print(f"--> {n_rows} reasons written to {out} "
          f"in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
