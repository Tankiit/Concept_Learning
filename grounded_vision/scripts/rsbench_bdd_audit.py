"""BDD-OIA (rsbench) premise-support audit — runs on the preprocessed
2048-d feature tensors; no images, no feature extraction.

Readers (logistic, 2048->21) -> concept bits -> per-action linear teacher +
depth-4 CART student over the 21 bits -> native TreeOps minimal sufficient
reasons -> premise support vs the 21 GT concept bits + A/B/C abstention arms.

Env: conda torch-multimodal (torch + sklearn + numpy).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

RUN = Path(__file__).parent
# Zenodo BDD-OIA-preprocessed bundle (gitignored scratch; rsbench_bdd_experiments.py
# takes --data-root if it moves)
ROOT = RUN.parent / "tmp" / "datasets" / "rsbench" / "bdd2048"
sys.path.insert(0, str(RUN.parent.parent / "bankexp"))
from utils.tree_ops import TreeOps  # noqa: E402

SEED = 42
N_CONCEPTS = 21
N_ACTIONS = 5
CONCEPT_NAMES = [
    "green_light", "follow", "clear", "red_light", "stop_sign", "car",
    "person", "rider", "other_obstacle", "left_lane", "left_green_light",
    "left_follow", "no_left_lane", "left_obstacle", "left_solid_line",
    "right_lane", "right_green_light", "right_follow", "no_right_lane",
    "right_obstacle", "right_solid_line",
]  # CONCEPTS_ORDER from rsbench-code rsseval, sorted by index
ACTION_NAMES = ["forward", "stop", "left", "right", "fifth_rare"]
# 4 active + one ~0.3% dim; names cosmetic — each bit audited as its own target


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_split(split):
    d = ROOT / split
    files = sorted(p.name for p in (d / "inputs").glob("*.pt"))
    X = np.zeros((len(files), 2048), dtype=np.float32)
    C = np.zeros((len(files), N_CONCEPTS), dtype=bool)
    Y = np.zeros((len(files), N_ACTIONS), dtype=bool)
    for i, f in enumerate(files):
        X[i] = torch.load(d / "inputs" / f, weights_only=True).numpy().ravel()
        C[i] = torch.load(d / "concepts" / f, weights_only=True).numpy().ravel() > 0.5
        Y[i] = torch.load(d / "labels" / f, weights_only=True).numpy().ravel() > 0.5
    return files, X, C, Y


def export_tree_json(clf, path):
    t = clf.tree_

    def rec(node):
        if t.children_left[node] == -1:
            return {"c": int(np.argmax(t.value[node]))}
        return {"f": int(t.feature[node]), "t": 0.5,
                "l": rec(t.children_left[node]), "r": rec(t.children_right[node])}

    Path(path).write_text(json.dumps(dict(
        classes=[0, 1], n_features=N_CONCEPTS, root=rec(0),
        feature_names=CONCEPT_NAMES)))
    return TreeOps(json.loads(Path(path).read_text()))


def main():
    rng = np.random.RandomState(SEED)
    data = {s: load_split(s) for s in ("train", "val", "test")}
    for s, (_, X, C, Y) in data.items():
        print(f"{s}: {len(X)} imgs, concept pos-rate mean {C.mean():.3f}, "
              f"actions/img {Y.sum(1).mean():.2f}")
    Xtr, Ctr, Ytr = data["train"][1], data["train"][2], data["train"][3]
    Xva, Cva = data["val"][1], data["val"][2]
    Xte, Cte, Yte = data["test"][1], data["test"][2], data["test"][3]

    # --- readers (21 logistics on their 2048-d features) ---
    readers, reader_va = [], []
    for j in range(N_CONCEPTS):
        y = Ctr[:, j].astype(int)
        if y.min() == y.max():
            readers.append((None, float(y[0]))); reader_va.append(1.0); continue
        m = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        m.fit(Xtr, y)
        readers.append((m, None))
        reader_va.append(float(m.score(Xva, Cva[:, j].astype(int))))
    print(f"readers: valid acc mean {np.mean(reader_va):.3f} "
          f"[{np.min(reader_va):.3f}, {np.max(reader_va):.3f}]")

    def probs(X):
        P = np.full((len(X), N_CONCEPTS), np.nan)
        for j, (m, const) in enumerate(readers):
            if m is not None:
                P[:, j] = m.predict_proba(X)[:, 1]
        return P

    P = {s: probs(data[s][1]) for s in data}
    B = {s: (P[s] >= 0.5) for s in data}

    # --- per-action teacher + tree, reasons, premise support ---
    per_action = {}
    all_reasons = []  # (action, idx, literals, all_true)
    for a in range(N_ACTIONS):
        ytr = Ytr[:, a].astype(int)
        if ytr.min() == ytr.max():
            continue
        teacher = LogisticRegression(max_iter=1000, random_state=SEED)
        teacher.fit(B["train"].astype(float), ytr)
        tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        tree.fit(B["train"].astype(float), ytr)
        ops = export_tree_json(tree, RUN / f"tree_bdd_a{a}.json")
        tp = teacher.predict(B["test"].astype(float))
        sp = np.array([ops.predict_one(x) for x in B["test"].astype(float)])
        rec = dict(
            base_rate=float(ytr.mean()),
            teacher_acc=float((tp == Yte[:, a].astype(int)).mean()),
            tree_acc=float((sp == Yte[:, a].astype(int)).mean()),
            agreement=float((tp == sp).mean()),
        )
        n_lit, all_true_a = [], []
        for k in range(len(B["test"])):
            if sp[k] != 1:          # certify only positive predictions
                continue
            reason = ops.minimal_reason(B["test"][k].astype(float), 1)
            lits = [(CONCEPT_NAMES[i], bool(v)) for i, v in reason]
            sup = [(Cte[k, CONCEPT_NAMES.index(nm)] == pol) for nm, pol in lits]
            n_lit.append(len(lits))
            ok = bool(all(sup))
            all_true_a.append(ok)
            all_reasons.append(dict(action=a, idx=k, pred=1,
                                    reason=[[nm, int(p)] for nm, p in lits],
                                    all_true=ok, n_true=int(sum(sup))))
        if all_true_a:
            p = float(np.mean(all_true_a)); lo, hi = wilson(p, len(all_true_a))
            rec.update(premise_support=p, ps_lo=lo, ps_hi=hi,
                       n_reasons=len(all_true_a),
                       reason_len_mean=float(np.mean(n_lit)))
        per_action[ACTION_NAMES[a]] = rec
        print(f"  {ACTION_NAMES[a]}: base {rec['base_rate']:.2f} "
              f"teacher {rec['teacher_acc']:.3f} tree {rec['tree_acc']:.3f} "
              f"agree {rec['agreement']:.3f} "
              + (f"P(all) {rec['premise_support']:.3f} n={rec['n_reasons']} "
                 f"len {rec['reason_len_mean']:.2f}" if "premise_support" in rec
                 else "(no positive test predictions)"))

    # --- pooled premise support + literal marginals ---
    at = np.array([r["all_true"] for r in all_reasons])
    p_pool = float(at.mean()); lo, hi = wilson(p_pool, len(at))
    print(f"POOLED premise support: {p_pool:.3f} [{lo:.3f}, {hi:.3f}] "
          f"n={len(at)} reasons (positive predictions)")

    lit_stats = {}
    for r in all_reasons:
        k = r["idx"]
        for nm, pol in [(l[0], bool(l[1])) for l in r["reason"]]:
            t = Cte[k, CONCEPT_NAMES.index(nm)] == pol
            d = lit_stats.setdefault((nm, pol), [0, 0])
            d[0] += 1; d[1] += int(t)
    marg = sorted(((k, v[1] / v[0], v[0]) for k, v in lit_stats.items()),
                  key=lambda x: -x[2] * (1 - x[1]))
    print("top failing (concept, polarity) slots:")
    for (nm, pol), s, c in [m for m in marg if m[2] >= 30][:12]:
        print(f"  {nm}={'1' if pol else '0'}: n={c}, support={s:.3f}")

    # --- abstention arms on pooled reasons ---
    from sklearn.isotonic import IsotonicRegression
    R = np.zeros((N_CONCEPTS, 2))
    for j in range(N_CONCEPTS):
        for b in (0, 1):
            mask = B["val"][:, j] == b
            R[j, b] = ((B["val"][mask, j] == Cva[mask, j]).mean()
                       if mask.sum() else 1.0)
    iso = []
    for j in range(N_CONCEPTS):
        ir = IsotonicRegression(out_of_bounds="clip")
        ok = np.isfinite(P["val"][:, j])
        ir.fit(P["val"][ok, j], Cva[ok, j].astype(int))
        iso.append(ir)

    def lit_score(r, kind):
        vals = []
        for nm, pol in [(l[0], bool(l[1])) for l in r["reason"]]:
            j = CONCEPT_NAMES.index(nm)
            if kind == "A":
                vals.append(R[j, int(pol)])
            elif kind == "B":
                pj = float(np.clip(P["test"][r["idx"], j], 1e-6, 1 - 1e-6))
                cal = float(iso[j].predict([pj])[0])
                vals.append(cal if pol else 1 - cal)
            else:
                pj = P["test"][r["idx"], j]
                vals.append(pj if pol else 1 - pj)
        return min(vals)

    n_all = len(all_reasons)
    curve = []
    for kind in ("A", "B", "C"):
        sc = np.array([lit_score(r, kind) for r in all_reasons])
        for cov in [1.0, 0.75, 0.5, 0.254, 0.136, 0.05]:
            n_cov = max(1, int(round(cov * n_all)))
            thr = np.partition(sc, -n_cov)[-n_cov]
            keep = sc >= thr
            p_cov = float(at[keep].mean()); lo_c, hi_c = wilson(p_cov, int(keep.sum()))
            rnd = float(np.mean([at[rng.choice(n_all, int(keep.sum()), replace=False)].mean()
                                 for _ in range(100)]))
            curve.append(dict(score=kind, coverage=float(keep.mean()),
                              p_all_given_cov=p_cov, lo=lo_c, hi=hi_c,
                              n=int(keep.sum()), random=rnd))
            print(f"  [{kind}] cov={keep.mean():.3f}: P(all|cov)={p_cov:.3f} "
                  f"[{lo_c:.3f},{hi_c:.3f}] n={int(keep.sum())} random={rnd:.3f}")

    out = dict(
        benchmark="rsbench BDD-OIA (bdd2048 preprocessed)",
        seed=SEED, n=dict(train=len(Xtr), val=len(Xva), test=len(Xte)),
        reader_valid_acc=[float(x) for x in reader_va],
        per_action=per_action,
        pooled_premise_support=dict(p=p_pool, lo=lo, hi=hi, n=n_all),
        literal_marginals={f"{nm}={int(pol)}": dict(n=c, support=float(s))
                           for (nm, pol), s, c in marg},
        abstention_curve=curve, reasons=all_reasons,
    )
    (RUN / "results").mkdir(exist_ok=True)
    (RUN / "results" / "bdd_oia_audit.json").write_text(json.dumps(out, indent=1))
    print("[done] results/bdd_oia_audit.json")


if __name__ == "__main__":
    main()
