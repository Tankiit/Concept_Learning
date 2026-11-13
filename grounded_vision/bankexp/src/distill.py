"""
distill.py — student tree distilled from the teacher's predictions.
=====================================================================
Stage 3 (PyXAI env or any env with sklearn).  CART on the TEACHER's
predictions over the bank inputs (Setting D / Design B protocol): the
tree is the deployed, exactly-explainable model; the linear teacher
stays frozen.

Usage
-----
    python src/distill.py --classes 4 --max-depth 4

Outputs (intermediate_results/)
    tree_K{K}.json           classes, feature_names, n_features, nested
                             root nodes -- consumed by src/explain.py,
                             analysis/*, utils/tree_ops.py
    tree_K{K}_metrics.json   test acc vs labels, teacher agreement,
                             depth, n_leaves
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"


def teacher_logits(X, teacher):
    return X @ teacher["W"].T + teacher["b"]


def teacher_predict(X, teacher):
    return teacher["classes"][np.argmax(teacher_logits(X, teacher), axis=1)]


def export_nodes(dt: DecisionTreeClassifier):
    t = dt.tree_

    def rec(nid):
        if t.children_left[nid] == -1 or t.feature[nid] < 0:
            return {"c": int(dt.classes_[int(np.argmax(t.value[nid][0]))])}
        return {"f": int(t.feature[nid]), "t": float(t.threshold[nid]),
                "l": rec(int(t.children_left[nid])),
                "r": rec(int(t.children_right[nid]))}

    return rec(0)


def distill(bank_dir, teacher_path, classes_k, max_depth=4,
            min_samples_leaf=4, seed=42, out_dir=None):
    out = Path(out_dir or bank_dir)
    tr = np.load(Path(bank_dir) / "bank_train.npz")
    te = np.load(Path(bank_dir) / "bank_test.npz")
    teacher = np.load(teacher_path)
    kind = str(teacher["input_kind"][0])
    meta = json.loads((Path(bank_dir) / "meta.json").read_text())
    tr_mask = tr["labels"] < classes_k
    te_mask = te["labels"] < classes_k
    Xtr = tr[kind].astype(np.float64)[tr_mask]
    Xte = te[kind].astype(np.float64)[te_mask]
    ytr = tr["labels"][tr_mask]
    yte = te["labels"][te_mask]
    ttr = teacher_predict(Xtr, teacher)
    tte = teacher_predict(Xte, teacher)
    dt = DecisionTreeClassifier(max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                random_state=seed).fit(Xtr, ttr)
    tree_json = {
        "classes": [int(c) for c in dt.classes_],
        "feature_names": meta["feature_names"],
        "n_features": len(meta["feature_names"]),
        "root": export_nodes(dt),
        "protocol": {"fit": "teacher predictions", "teacher_input": kind,
                     "max_depth": max_depth,
                     "min_samples_leaf": min_samples_leaf,
                     "seed": seed, "K": int(classes_k)},
    }
    (out / f"tree_K{classes_k}.json").write_text(json.dumps(tree_json))
    pred = dt.predict(Xte)
    metrics = {
        "K": int(classes_k), "n_test": int(len(yte)),
        "test_acc": round(float(np.mean(pred == yte)), 4),
        "teacher_test_acc": round(float(np.mean(tte == yte)), 4),
        "teacher_agreement": round(float(np.mean(pred == tte)), 4),
        "depth": int(dt.get_depth()), "n_leaves": int(dt.get_n_leaves()),
    }
    (out / f"tree_K{classes_k}_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    print(f"--> distilled tree K={classes_k}: test acc {metrics['test_acc']} | "
          f"teacher agreement {metrics['teacher_agreement']} | "
          f"depth {metrics['depth']} leaves {metrics['n_leaves']}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=4)
    ap.add_argument("--out", default=str(INTERMEDIATE_DIR))
    args = ap.parse_args()
    teacher = args.teacher or str(
        Path(args.bank) / f"teacher_K{args.classes}.npz")
    distill(args.bank, teacher, args.classes, max_depth=args.max_depth,
            min_samples_leaf=args.min_leaf, out_dir=args.out)


if __name__ == "__main__":
    main()
