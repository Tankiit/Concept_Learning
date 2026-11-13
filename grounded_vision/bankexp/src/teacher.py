"""
teacher.py — Design B teacher: linear CBM head over the concept bank.
======================================================================
Stage 2 (PyC env: conda).  Multinomial logistic head on the bank
inputs (--on bits: thresholded predicates, the deployed-protocol
default; --on scores: continuous concept scores, the standard-CBM
variant).  Linear over [0,1]-box inputs is what makes the exact
box-LP certification in analysis/margin_sufficiency.py possible.

Usage
-----
    python src/teacher.py --classes 4 --on bits

Outputs (intermediate_results/)
    teacher_K{K}.npz         W (K x F), b (K,), classes (K,), input_kind
    teacher_K{K}_metrics.json  train/test accuracy, majority baseline
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"


def fit_teacher(bank_dir, classes_k, input_kind="bits", seed=42, out_dir=None):
    out = Path(out_dir or bank_dir)
    tr = np.load(Path(bank_dir) / "bank_train.npz")
    te = np.load(Path(bank_dir) / "bank_test.npz")
    Xtr_all = tr[input_kind].astype(np.float64)
    Xte_all = te[input_kind].astype(np.float64)
    tr_mask = tr["labels"] < classes_k
    te_mask = te["labels"] < classes_k
    Xtr, ytr = Xtr_all[tr_mask], tr["labels"][tr_mask]
    Xte, yte = Xte_all[te_mask], te["labels"][te_mask]
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(Xtr, ytr)
    np.savez(out / f"teacher_K{classes_k}.npz",
             W=clf.coef_.astype(np.float64),
             b=clf.intercept_.astype(np.float64),
             classes=clf.classes_.astype(np.int64),
             input_kind=np.array([input_kind]))
    metrics = {
        "K": int(classes_k), "input_kind": input_kind,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_acc": round(float(clf.score(Xtr, ytr)), 4),
        "test_acc": round(float(clf.score(Xte, yte)), 4),
        "majority_acc": round(float(np.mean(
            yte == np.bincount(ytr).argmax())), 4),
    }
    (out / f"teacher_K{classes_k}_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    print(f"--> teacher({input_kind}, K={classes_k}): "
          f"test acc {metrics['test_acc']} "
          f"(majority {metrics['majority_acc']})", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--on", choices=["bits", "scores"], default="bits")
    ap.add_argument("--out", default=str(INTERMEDIATE_DIR))
    args = ap.parse_args()
    fit_teacher(args.bank, args.classes, input_kind=args.on, out_dir=args.out)


if __name__ == "__main__":
    main()
