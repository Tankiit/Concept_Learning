"""Stage C: CelebA premise-support audit (Smiling target, 39-concept vocabulary).

Readers (logistic on DINOv2 features) -> 40 concept bits -> linear teacher on
39 non-target bits -> sklearn CART depth 4 -> native TreeOps minimal sufficient
reasons -> premise support vs the 40 GT attribute bits.

Runs in conda torch-multimodal (numpy + sklearn). Writes results/ JSON + MD.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

RUN = Path(__file__).parent
sys.path.insert(0, str(RUN.parent.parent / "bankexp"))
from utils.tree_ops import TreeOps  # noqa: E402

TARGET = "Smiling"
SEED = 42


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def export_tree_json(clf, feature_names, path):
    t = clf.tree_

    def rec(node):
        if t.children_left[node] == -1:
            return {"c": int(np.argmax(t.value[node]))}
        return {
            "f": int(t.feature[node]),
            "t": 0.5,
            "l": rec(t.children_left[node]),
            "r": rec(t.children_right[node]),
        }

    obj = {
        "classes": [0, 1],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "root": rec(0),
    }
    Path(path).write_text(json.dumps(obj))
    return obj


def main():
    rng = np.random.RandomState(SEED)
    cols = (RUN / "train_columns.txt").read_text().splitlines()
    attrs = {s: np.load(RUN / f"{s}_attrs.npy") for s in ("train", "valid", "test")}
    feats = {s: np.load(RUN / f"feats_{s}.npy") for s in ("train", "valid", "test")}
    print(f"splits: " + ", ".join(f"{s}={len(v)}" for s, v in attrs.items()))

    # --- readers: one logistic per attribute on DINOv2 features ---
    readers, reader_acc = {}, {}
    for j, name in enumerate(cols):
        y = attrs["train"][:, j].astype(int)
        if y.min() == y.max():
            readers[name] = (None, float(y[0]))
            continue
        m = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        m.fit(feats["train"], y)
        acc = float(m.score(feats["valid"], attrs["valid"][:, j].astype(int)))
        readers[name] = (m, None)
        reader_acc[name] = acc
    accs = np.array(list(reader_acc.values()))
    print(f"readers: {len(readers)} fitted, valid acc mean {accs.mean():.3f} "
          f"[{accs.min():.3f}, {accs.max():.3f}]")

    def probs(split):
        n = len(attrs[split])
        P = np.full((n, len(cols)), np.nan)
        for j, name in enumerate(cols):
            m, const = readers[name]
            if m is not None:
                P[:, j] = m.predict_proba(feats[split])[:, 1]
        return P

    Probs = {s: probs(s) for s in ("train", "valid", "test")}
    Bits = {s: (Probs[s] >= 0.5) for s in ("train", "valid", "test")}
    gt = {s: attrs[s] for s in attrs}
    ti = cols.index(TARGET)
    vocab = [c for c in cols if c != TARGET]
    vi = [cols.index(c) for c in vocab]

    # --- teacher: linear head over the 39 non-target bits ---
    teacher = LogisticRegression(max_iter=1000, random_state=SEED)
    teacher.fit(Bits["train"][:, vi].astype(float), attrs["train"][:, ti].astype(int))
    teach_pred = {s: teacher.predict(Bits[s][:, vi].astype(float))
                  for s in ("train", "valid", "test")}

    # --- student: CART depth 4 over the same 39 bits ---
    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    tree.fit(Bits["train"][:, vi].astype(float), attrs["train"][:, ti].astype(int))
    tree_path = RUN / "tree_celeba_smiling.json"
    export_tree_json(tree, vocab, tree_path)
    ops = TreeOps(json.loads(tree_path.read_text()))

    res = {}
    for s in ("valid", "test"):
        y = attrs[s][:, ti].astype(int)
        tp = teach_pred[s]
        sp = np.array([ops.predict_one(x) for x in Bits[s][:, vi].astype(float)])
        res[s] = dict(
            teacher_acc=float((tp == y).mean()),
            tree_acc=float((sp == y).mean()),
            agreement=float((tp == sp).mean()),
        )
    print(f"test: teacher {res['test']['teacher_acc']:.3f}, tree {res['test']['tree_acc']:.3f}, "
          f"agree {res['test']['agreement']:.3f}")

    # --- reasons + premise support (test split) ---
    per_instance = []
    n_lits_all = []
    for k in range(len(Bits["test"])):
        x = Bits["test"][k, vi].astype(float)
        pred = int(ops.predict_one(x))
        reason = ops.minimal_reason(x, pred)  # list of (feat_idx, value) pairs
        lits = [(vocab[i], bool(v)) for i, v in reason]
        n_lits_all.append(len(lits))
        support = [(bool(gt["test"][k, cols.index(nm)]) == pol) for nm, pol in lits]
        per_instance.append(dict(
            idx=k, pred=pred, gt=int(gt["test"][k, ti]),
            reason=[[nm, int(p)] for nm, p in lits],
            all_true=bool(all(support)), n_true=int(sum(support)),
        ))
    P_all = float(np.mean([p["all_true"] for p in per_instance]))
    n = len(per_instance)
    lo, hi = wilson(P_all, n)
    print(f"premise support P(all|reason): {P_all:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
    print(f"reason length: mean {np.mean(n_lits_all):.2f}, max {np.max(n_lits_all)}")

    # per-(concept, polarity) marginal support among in-reason literals
    lit_stats = {}
    for p in per_instance:
        # recompute per-literal truth
        k = p["idx"]
        for nm, pol in [(l[0], bool(l[1])) for l in p["reason"]]:
            key = (nm, pol)
            t = bool(gt["test"][k, cols.index(nm)]) == pol
            d = lit_stats.setdefault(key, [0, 0])  # [n, n_true]
            d[0] += 1
            d[1] += int(t)
    marg = sorted(((k, v[1] / v[0], v[0]) for k, v in lit_stats.items()),
                  key=lambda x: -x[2] * (1 - x[1]))
    print("top failing (concept, polarity) slots [n_in_reason, support]:")
    for (nm, pol), s, cnt in [m for m in marg if m[2] >= 30][:12]:
        print(f"  {nm}={'1' if pol else '0'}: n={cnt}, support={s:.3f}")

    # --- abstention arm: val-estimated per-(attr, polarity) literal reliability ---
    # R[a, b] = P(reader bit == GT | reader bit = b) for attribute a, estimated on
    # the valid split (deployment-honest, no test labels); abstain when the minimum
    # reliability over a reason's literals is below t; sweep coverage.
    R = np.zeros((len(cols), 2))
    for j in range(len(cols)):
        for b in (0, 1):
            mask = Bits["valid"][:, j] == b
            R[j, b] = ((Bits["valid"][mask, j] == gt["valid"][mask, j]).mean()
                       if mask.sum() else 1.0)
    scores = np.array([min((R[cols.index(nm), int(pol)] for nm, pol in
                            [(l[0], bool(l[1])) for l in p["reason"]]))
                       for p in per_instance])
    all_true = np.array([p["all_true"] for p in per_instance])

    def sweep(score_name, scores, curve_out):
        for cov in [1.0, 0.9, 0.75, 0.5, 0.254, 0.136, 0.05]:
            n_cov = max(1, int(round(cov * n_all)))
            thr = np.partition(scores, -n_cov)[-n_cov]
            keep = scores >= thr
            n_keep = int(keep.sum())
            p_cov = float(all_true[keep].mean())
            lo_c, hi_c = wilson(p_cov, n_keep)
            rnd = float(np.mean([all_true[rng.choice(n_all, n_keep, replace=False)].mean()
                                 for _ in range(100)]))
            curve_out.append(dict(score=score_name, coverage=n_keep / n_all,
                                  p_all_given_cov=p_cov, lo=lo_c, hi=hi_c,
                                  n=n_keep, random=rnd))
            print(f"  [{score_name}] cov={n_keep / n_all:.3f}: P(all|cov)={p_cov:.3f} "
                  f"[{lo_c:.3f},{hi_c:.3f}] n={n_keep}  random={rnd:.3f}")

    n_all = len(all_true)
    curve = []
    # A_discrete: min val-estimated P(bit correct | attr, polarity, bit)
    sweep("A_discrete", scores, curve)
    # B_cal: min val-isotonic per-attr calibrated P(asserted bit true)
    from sklearn.isotonic import IsotonicRegression
    B_scores = np.zeros(n_all)
    iso = {}
    for j, name in enumerate(cols):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(Probs["valid"][:, j], gt["valid"][:, j].astype(int))
        iso[name] = ir
    for k, p in enumerate(per_instance):
        vals = []
        for nm, pol in [(l[0], bool(l[1])) for l in p["reason"]]:
            pj = float(np.clip(Probs["test"][k, cols.index(nm)], 1e-6, 1 - 1e-6))
            cal = float(iso[nm].predict([pj])[0])
            vals.append(cal if pol else 1.0 - cal)
        B_scores[k] = min(vals)
    sweep("B_cal", B_scores, curve)
    # C_raw: min raw reader confidence in the asserted bit
    C_scores = np.array([min((Probs["test"][k, cols.index(nm)] if pol else
                              1 - Probs["test"][k, cols.index(nm)])
                             for nm, pol in [(l[0], bool(l[1])) for l in p["reason"]])
                         for k, p in enumerate(per_instance)])
    sweep("C_raw", C_scores, curve)

    out = dict(
        target=TARGET, backbone="dinov2_vits14", seed=SEED,
        n=dict(train=len(attrs['train']), valid=len(attrs['valid']), test=len(attrs['test'])),
        reader_valid_acc_mean=float(accs.mean()),
        reader_valid_acc={k: float(v) for k, v in reader_acc.items()},
        head=res["test"], premise_support=dict(p=P_all, lo=lo, hi=hi, n=n),
        reason_len=dict(mean=float(np.mean(n_lits_all)), max=int(np.max(n_lits_all))),
        literal_marginals={f"{nm}={int(pol)}": dict(n=c, support=float(s))
                           for (nm, pol), s, c in marg},
        abstention_curve=curve,
        per_instance=per_instance,
    )
    (RUN / "results").mkdir(exist_ok=True)
    (RUN / "results" / "celeba_audit_smiling.json").write_text(json.dumps(out, indent=1))
    print("[done] results/celeba_audit_smiling.json")


if __name__ == "__main__":
    main()
