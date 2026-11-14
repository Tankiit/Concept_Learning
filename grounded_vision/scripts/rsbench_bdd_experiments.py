"""BDD-OIA (rsbench) experiment suite — the CUB keep-list rungs on the
second driving dataset, beyond the premise-support audit.

Stages (seed 42, official splits, their preprocessed 2048-d tensors):
  1. reproduce the committed audit pipeline (logistic readers -> bits ->
     per-action linear teacher + depth-4 CART -> native minimal reasons)
     and check pooled P(all) against results/bdd_oia_audit.json;
  2. independence gap + Q8 literal filter + Q4 polarity-vs-base +
     failure concentration (pooled and per action), cluster bootstrap
     (vectorised) + 2-fold cross-fit;
  3. escape attempts on the frozen trees: oracle subset selection
     (exhaustive over <=2^4 path subsets), UAR-vs-ORD max-q reasons
     (raw + calibrated arms), trusted-only fixed-delta rules (global
     concept accuracy / per-instance calibrated confidence), single-
     clause and negative-only survivors;
  4. certified-only retrain (teacher+tree on train instances whose train
     reason is fully GT-supported);
  5. stronger-reader null (torch MLP readers instead of logistic);
  6. ladder rungs: direct 2048->action probe, deployed bits, GT-bits
     teacher (vocabulary ceiling), GT-bits tree (collapse check);
  7. label-noise positive control (iid GT-bit flips, eps sweep).

Output: results/bdd_oia_experiments.json (rewritten after every stage).

Env: ltn_pyxai_starter/.venv (torch 2.8 + sklearn 1.8 + numpy).
Data: tmp/datasets/rsbench/bdd2048 (Zenodo BDD-OIA-preprocessed; override
with --data-root).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

GV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GV / "bankexp"))
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
]
ACTION_NAMES = ["forward", "stop", "left", "right", "fifth_rare"]
ACTIVE_ACTIONS = [0, 1, 2, 3]  # fifth_rare: base 0.003, no positive preds


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_split(root, split):
    d = root / split
    files = sorted(p.name for p in (d / "inputs").glob("*.pt"))
    X = np.zeros((len(files), 2048), dtype=np.float32)
    C = np.zeros((len(files), N_CONCEPTS), dtype=bool)
    Y = np.zeros((len(files), N_ACTIONS), dtype=bool)
    for i, f in enumerate(files):
        X[i] = torch.load(d / "inputs" / f, weights_only=True).numpy().ravel()
        C[i] = torch.load(d / "concepts" / f, weights_only=True).numpy().ravel() > 0.5
        Y[i] = torch.load(d / "labels" / f, weights_only=True).numpy().ravel() > 0.5
    return files, X, C, Y


def tree_ops_from(clf):
    t = clf.tree_

    def rec(node):
        if t.children_left[node] == -1:
            return {"c": int(np.argmax(t.value[node]))}
        return {"f": int(t.feature[node]), "t": 0.5,
                "l": rec(t.children_left[node]), "r": rec(t.children_right[node])}

    return TreeOps(dict(classes=[0, 1], n_features=N_CONCEPTS, root=rec(0),
                        feature_names=CONCEPT_NAMES))


def fit_teacher_tree(Btr, ytr, seed=SEED):
    teacher = LogisticRegression(max_iter=1000, random_state=seed)
    teacher.fit(Btr.astype(float), ytr)
    tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
    tree.fit(Btr.astype(float), ytr)
    return teacher, tree, tree_ops_from(tree)


def reasons_on(ops, B, C, action, idxs):
    """Minimal reasons + per-literal GT support for positive predictions."""
    out = []
    for k in idxs:
        if ops.predict_one(B[k].astype(float)) != 1:
            continue
        reason = ops.minimal_reason(B[k].astype(float), 1)
        lits = [(int(f), int(b)) for f, b in reason]
        sup = [bool(C[k, f] == b) for f, b in lits]
        out.append(dict(action=action, idx=int(k), lits=lits, sup=sup,
                        all_true=bool(all(sup))))
    return out


# ---------------------------------------------------------------- stage 2
def block_gap(rows, label, n_boot, rng):
    """Independence-gap analysis for one block of reasons (vectorised)."""
    n = len(rows)
    L = max(len(r["lits"]) for r in rows)
    ids = np.zeros((n, L), dtype=np.int64)
    sup = np.zeros((n, L), dtype=bool)
    valid = np.zeros((n, L), dtype=bool)
    vocab = {}
    obs = np.zeros(n, dtype=bool)
    lens = np.zeros(n, dtype=int)
    for i, r in enumerate(rows):
        lens[i] = len(r["lits"])
        obs[i] = r["all_true"]
        for j, ((f, b), s) in enumerate(zip(r["lits"], r["sup"])):
            key = f"{CONCEPT_NAMES[f]}|{b}"
            if key not in vocab:
                vocab[key] = len(vocab)
            ids[i, j] = vocab[key]
            sup[i, j] = s
            valid[i, j] = True
    n_lit = len(vocab)

    def marginals(id_a, sup_a):
        cnt = np.bincount(id_a, minlength=n_lit)
        s = np.bincount(id_a, weights=sup_a.astype(float), minlength=n_lit)
        return s / np.clip(cnt, 1, None)

    flat_ids, flat_sup = ids[valid], sup[valid]
    p_lit = marginals(flat_ids, flat_sup)
    prod = np.where(valid, p_lit[ids], 1.0)
    pred = prod.prod(1)

    boots = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        ib, vb, sb = ids[idx], valid[idx], sup[idx]
        pb = marginals(ib[vb], sb[vb])
        pred_b = np.where(vb, pb[ib], 1.0).prod(1)
        boots[b] = (obs[idx].mean(), pred_b.mean())
    ratio_boot = boots[:, 0] / boots[:, 1]

    # 2-fold cross-fit (marginals estimated out-of-half, pooled fallback
    # for literal ids unseen in the estimation half)
    perm = rng.permutation(n)
    halves = [perm[: n // 2], perm[n // 2:]]
    pred_x = np.empty(n)
    for est, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
        ie, ve, se = ids[est], valid[est], sup[est]
        cnt_e = np.bincount(ie[ve], minlength=n_lit)
        pe = marginals(ie[ve], se[ve])
        pool = se[ve].mean()
        pm = np.where(cnt_e > 0, pe, pool)
        pred_x[ev] = np.where(valid[ev], pm[ids[ev]], 1.0).prod(1)

    hist = dict(n_distinct_literals=int(n_lit),
                at_0=float((p_lit == 0).mean()),
                at_1=float((p_lit == 1).mean()),
                strictly_between=float(((p_lit > 0) & (p_lit < 1)).mean()))

    def filt(keep, t):
        cov = float(keep.mean())
        return dict(threshold=t, coverage=cov, n_covered=int(keep.sum()),
                    p_all_given_covered=float(obs[keep].mean()) if keep.any() else None)

    q8, q8x = {}, {}
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        keep = valid.all(1) & (np.where(valid, p_lit[ids], 1.0).min(1) >= t)
        q8[str(t)] = filt(keep, t)
        covered_x = np.zeros(n, dtype=bool)
        for est, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
            ie, ve, se = ids[est], valid[est], sup[est]
            cnt_e = np.bincount(ie[ve], minlength=n_lit)
            pe = marginals(ie[ve], se[ve])
            pm = np.where(cnt_e > 0, pe, se[ve].mean())
            covered_x[ev] = valid[ev].all() and np.where(
                valid[ev], pm[ids[ev]], 1.0).min() >= t
        q8x[str(t)] = filt(covered_x, t)

    out = dict(
        block=label, n=n, mean_len=float(lens.mean()),
        observed_joint=float(obs.mean()),
        independence_predicted_joint=float(pred.mean()),
        ratio_obs_over_pred=float(obs.mean() / pred.mean()),
        ratio_boot_pct=[float(np.percentile(ratio_boot, 2.5)),
                        float(np.percentile(ratio_boot, 97.5))],
        ratio_crossfit=float(obs.mean() / pred_x.mean()),
        literal_marginal_hist=hist, q8_literal_filter=q8,
        q8_literal_filter_crossfit=q8x, by_length={},
    )
    for lo in range(1, int(lens.max()) + 1):
        m = lens == lo
        if m.sum():
            out["by_length"][f"L{lo}"] = dict(
                n=int(m.sum()), observed=float(obs[m].mean()),
                predicted=float(pred[m].mean()))
    return out


def q4_and_concentration(rows, C):
    """Polarity precision vs base rates + failing-slot concentration."""
    occ = []
    for r in rows:
        for (f, b), s in zip(r["lits"], r["sup"]):
            occ.append((f, b, bool(s)))
    occ = np.array(occ, dtype=object)
    bits = np.array([o[1] for o in occ])
    sup = np.array([o[2] for o in occ])
    feats = np.array([o[0] for o in occ])
    base = {f: float((C[:, f] == 1).mean()) for f in np.unique(feats)}
    present = bits == 1
    pres_base = np.mean([base[f] for f in feats[present]])
    abs_base = np.mean([1 - base[f] for f in feats[~present]])

    slot = {}
    for f in np.unique(feats):
        for b in (0, 1):
            m = (feats == f) & (bits == b)
            if m.sum() >= 30:
                p = float(sup[m].mean())
                slot[f"{CONCEPT_NAMES[f]}={b}"] = dict(
                    n=int(m.sum()), support=p)
    ranked = sorted(slot.items(), key=lambda kv: -kv[1]["n"] * (1 - kv[1]["support"]))
    failing = [r for r in rows if not r["all_true"]]
    top3 = [k for k, _ in ranked[:3]]
    contains_top1 = np.mean([any(f"{CONCEPT_NAMES[f]}={b}" == ranked[0][0]
                                for f, b in r["lits"]) for r in failing])
    contains_top3 = np.mean([any(f"{CONCEPT_NAMES[f]}={b}" in top3
                                 for f, b in r["lits"]) for r in failing])
    return dict(
        n_occurrences=int(len(occ)),
        present_precision=float(sup[present].mean()),
        present_base_rate=float(pres_base),
        absent_precision=float(sup[~present].mean()),
        absent_base_rate=float(abs_base),
        top_failing_slots=ranked[:8],
        frac_failing_reasons_with_top1_slot=float(contains_top1),
        frac_failing_reasons_with_any_top3_slot=float(contains_top3),
        n_failing=len(failing),
    )


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(GV / "tmp" / "datasets" / "rsbench" / "bdd2048"))
    ap.add_argument("--stages", default="1,2,3,4,5,6,7")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(GV / "results" / "bdd_oia_experiments.json"))
    args = ap.parse_args()
    stages = {int(s) for s in args.stages.split(",")}
    root = Path(args.data_root)
    rng = np.random.RandomState(SEED)
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    payload = dict(
        benchmark="rsbench BDD-OIA (bdd2048 preprocessed)", seed=SEED,
        protocol=dict(
            splits="official 16082/2270/4572, their preprocessed 2048-d tensors",
            reader="logistic per concept (stage 5: torch MLP 2048-256-1)",
            teacher="per-action logistic on 21 bits", tree="CART depth 4",
            reasons="native TreeOps backward-elimination minimal sufficient",
            bootstrap=f"cluster over reasons, {args.boot} resamples, "
                      "marginals re-estimated per resample",
        ),
    )

    def save():
        out_path.write_text(json.dumps(payload, indent=1))

    t0 = time.time()
    print("loading tensors ...", flush=True)
    data = {s: load_split(root, s) for s in ("train", "val", "test")}
    Xtr, Ctr, Ytr = data["train"][1], data["train"][2], data["train"][3]
    Xva, Cva = data["val"][1], data["val"][2]
    Xte, Cte, Yte = data["test"][1], data["test"][2], data["test"][3]
    print(f"  loaded in {time.time() - t0:.0f}s: "
          f"{len(Xtr)}/{len(Xva)}/{len(Xte)}", flush=True)

    # --- readers (shared by stages 1-4, 6) ---
    print("fitting logistic readers ...", flush=True)
    readers = []
    for j in range(N_CONCEPTS):
        y = Ctr[:, j].astype(int)
        if y.min() == y.max():
            readers.append((None, float(y[0])))
            continue
        m = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        m.fit(Xtr, y)
        readers.append((m, None))

    def probs(X):
        P = np.full((len(X), N_CONCEPTS), np.nan)
        for j, (m, const) in enumerate(readers):
            if m is not None:
                P[:, j] = m.predict_proba(X)[:, 1]
            else:
                P[:, j] = const
        return P

    P = {s: probs(data[s][1]) for s in data}
    B = {s: (P[s] >= 0.5) for s in data}
    reader_va = [float((B["val"][:, j] == Cva[:, j]).mean())
                 for j in range(N_CONCEPTS)]
    print(f"readers: valid acc mean {np.mean(reader_va):.3f}", flush=True)

    # per-action teacher+tree (stage-1 pipeline, reused by 3 and 6)
    fitted = {}
    for a in ACTIVE_ACTIONS:
        ytr = Ytr[:, a].astype(int)
        fitted[a] = fit_teacher_tree(B["train"], ytr)

    # ------------------------------------------------------------- stage 1
    if 1 in stages:
        rows_all, per_action = [], {}
        for a in ACTIVE_ACTIONS:
            teacher, tree, ops = fitted[a]
            yte = Yte[:, a].astype(int)
            tp = teacher.predict(B["test"].astype(float))
            sp = np.array([ops.predict_one(x) for x in B["test"].astype(float)])
            rows = reasons_on(ops, B["test"], Cte, a, range(len(Xte)))
            at = np.array([r["all_true"] for r in rows])
            p, (lo, hi) = float(at.mean()), wilson(at.mean(), len(at))
            per_action[ACTION_NAMES[a]] = dict(
                base_rate=float(ytr.mean()),
                teacher_acc=float((tp == yte).mean()),
                tree_acc=float((sp == yte).mean()),
                agreement=float((tp == sp).mean()),
                p_all=p, lo=lo, hi=hi, n=len(rows),
                reason_len_mean=float(np.mean([len(r["lits"]) for r in rows])))
            rows_all.extend(rows)
            print(f"  {ACTION_NAMES[a]}: P(all) {p:.3f} [{lo:.3f},{hi:.3f}] "
                  f"n={len(rows)}", flush=True)
        at = np.array([r["all_true"] for r in rows_all])
        p, (lo, hi) = float(at.mean()), wilson(at.mean(), len(at))
        repro = None
        committed = GV / "results" / "bdd_oia_audit.json"
        if committed.exists():
            c = json.loads(committed.read_text())
            repro = dict(committed_pooled=c["pooled_premise_support"]["p"],
                         this_run=p, delta=p - c["pooled_premise_support"]["p"],
                         committed_n=c["pooled_premise_support"]["n"], this_n=len(rows_all))
        payload["stage1_reproduce"] = dict(
            reader_valid_acc=[float(x) for x in reader_va],
            per_action=per_action,
            pooled=dict(p=p, lo=lo, hi=hi, n=len(rows_all)),
            check_vs_committed=repro)
        print(f"pooled P(all) {p:.4f} [{lo:.3f},{hi:.3f}] n={len(rows_all)}; "
              f"committed delta {repro['delta']:+.4f}" if repro else "", flush=True)
        save()

    # rebuild rows for later stages if stage 1 skipped
    if any(s in stages for s in (2, 3, 7)):
        if "stage1_reproduce" not in payload:
            rows_all = []
            for a in ACTIVE_ACTIONS:
                _, _, ops = fitted[a]
                rows_all.extend(reasons_on(ops, B["test"], Cte, a,
                                           range(len(Xte))))

    # ------------------------------------------------------------- stage 2
    if 2 in stages:
        print("stage 2: independence gap ...", flush=True)
        blocks = [block_gap(rows_all, "pooled", args.boot, rng)]
        for a in ACTIVE_ACTIONS:
            rows_a = [r for r in rows_all if r["action"] == a]
            blocks.append(block_gap(rows_a, ACTION_NAMES[a], args.boot, rng))
        for b in blocks:
            ci = b["ratio_boot_pct"]
            print(f"  {b['block']:<8} n={b['n']:<5} len {b['mean_len']:.2f} "
                  f"obs {b['observed_joint']:.3f} pred {b['independence_predicted_joint']:.3f} "
                  f"ratio {b['ratio_obs_over_pred']:.3f} [{ci[0]:.2f},{ci[1]:.2f}] "
                  f"xfit {b['ratio_crossfit']:.3f}", flush=True)
        payload["stage2_independence_gap"] = dict(
            blocks=blocks, q4_concentration=q4_and_concentration(rows_all, Cte))
        q = payload["stage2_independence_gap"]["q4_concentration"]
        print(f"  present prec {q['present_precision']:.3f} vs base "
              f"{q['present_base_rate']:.3f} | absent prec {q['absent_precision']:.3f} "
              f"vs base {q['absent_base_rate']:.3f}", flush=True)
        print(f"  failing reasons containing top-1 slot: "
              f"{q['frac_failing_reasons_with_top1_slot']:.3f} "
              f"(top slot {q['top_failing_slots'][0][0]}, n_failing={q['n_failing']})",
              flush=True)
        save()

    # ------------------------------------------------------------- stage 3
    if 3 in stages:
        print("stage 3: escape attempts ...", flush=True)
        iso = []
        for j in range(N_CONCEPTS):
            ir = IsotonicRegression(out_of_bounds="clip")
            ok = np.isfinite(P["val"][:, j])
            ir.fit(P["val"][ok, j], Cva[ok, j].astype(int))
            iso.append(ir)
        val_acc = reader_va

        def p_true(f, b, k):
            cal = float(iso[f].predict([P["test"][k, f]])[0])
            return cal if b else 1 - cal

        agg = dict(n=0, ord_all=0, oracle_any=0, oracle_max_supfrac=[],
                   uar_eq_ord=0, q_ord_cal=[], q_uar_cal=[], q_ord_raw=[],
                   q_uar_raw=[], len_ord=[], len_uar=[], len1=0, len1_all=0,
                   neg_only=0, neg_only_all=0, any_pos=0, any_pos_all=0,
                   ord_minconf_lt08=0, ord_minconf_lt05=0)
        deltas = (0.6, 0.65, 0.7, 0.75, 0.8)
        trust = {d: dict(g_cov=0, g_kept=0, g_ok=0, i_cov=0, i_kept=0, i_ok=0)
                 for d in deltas}
        for r in rows_all:
            a, k = r["action"], r["idx"]
            _, _, ops = fitted[a]
            path = ops.path_literals(B["test"][k].astype(float))
            path = [(int(f), int(b)) for f, b in path]
            suff = []
            for mask in range(1 << len(path)):
                sub = [path[i] for i in range(len(path)) if mask >> i & 1]
                if ops.forces({f: b for f, b in sub}, 1):
                    suff.append(sub)
            ordlits = [(f, b) for f, b in
                       ops.minimal_reason(B["test"][k].astype(float), 1)]
            subs = []
            for sub in suff:
                sups = [bool(Cte[k, f] == b) for f, b in sub]
                qc = np.prod([p_true(f, b, k) for f, b in sub]) if sub else 1.0
                qr = np.prod([P["test"][k, f] if b else 1 - P["test"][k, f]
                              for f, b in sub]) if sub else 1.0
                subs.append((sub, all(sups), float(np.mean(sups)), qc, qr))
            oracle_any = any(s[1] for s in subs)
            best_qc = max(subs, key=lambda s: (s[3], -len(s[0])))
            best_qr = max(subs, key=lambda s: (s[4], -len(s[0])))
            ordq_c = float(np.prod([p_true(f, b, k) for f, b in ordlits]))
            ordq_r = float(np.prod([P["test"][k, f] if b else 1 - P["test"][k, f]
                                    for f, b in ordlits]))
            agg["n"] += 1
            agg["ord_all"] += r["all_true"]
            agg["oracle_any"] += oracle_any
            agg["oracle_max_supfrac"].append(max(s[2] for s in subs))
            agg["uar_eq_ord"] += int(set(best_qc[0]) == set(ordlits))
            agg["q_ord_cal"].append(ordq_c)
            agg["q_uar_cal"].append(best_qc[3])
            agg["q_ord_raw"].append(ordq_r)
            agg["q_uar_raw"].append(best_qr[4])
            agg["len_ord"].append(len(ordlits))
            agg["len_uar"].append(len(best_qc[0]))
            if len(ordlits) == 1:
                agg["len1"] += 1
                agg["len1_all"] += r["all_true"]
            if all(b == 0 for f, b in ordlits):
                agg["neg_only"] += 1
                agg["neg_only_all"] += r["all_true"]
            else:
                agg["any_pos"] += 1
                agg["any_pos_all"] += r["all_true"]
            confs = [p_true(f, b, k) for f, b in ordlits]
            agg["ord_minconf_lt08"] += int(min(confs) < 0.8)
            agg["ord_minconf_lt05"] += int(min(confs) < 0.5)
            for d in deltas:
                kg = [(f, b) for f, b in ordlits if val_acc[f] >= d]
                ki = [(f, b) for f, b in ordlits if p_true(f, b, k) >= d]
                if ops.forces({f: b for f, b in kg}, 1):
                    trust[d]["g_cov"] += 1
                    trust[d]["g_kept"] += len(kg)
                    trust[d]["g_ok"] += all(Cte[k, f] == b for f, b in kg)
                if ops.forces({f: b for f, b in ki}, 1):
                    trust[d]["i_cov"] += 1
                    trust[d]["i_kept"] += len(ki)
                    trust[d]["i_ok"] += all(Cte[k, f] == b for f, b in ki)
        # full-vector trusted-only (CUB guarantee_scaling protocol: keep every
        # concept literal of the image whose trust >= delta, over all test
        # images; covered = kept set alone forces the image's prediction)
        fv = {d: dict(g_cov=0, g_kept=0, g_ok=0, i_cov=0, i_kept=0, i_ok=0,
                      n=0) for d in deltas}
        for a in ACTIVE_ACTIONS:
            _, _, ops = fitted[a]
            for k in range(len(Xte)):
                pred = ops.predict_one(B["test"][k].astype(float))
                fv_bits = [(j, int(B["test"][k, j])) for j in range(N_CONCEPTS)]
                for d in deltas:
                    fv[d]["n"] += 1
                    kg = [(j, b) for j, b in fv_bits if val_acc[j] >= d]
                    ki = [(j, b) for j, b in fv_bits if p_true(j, b, k) >= d]
                    if ops.forces({j: b for j, b in kg}, pred):
                        fv[d]["g_cov"] += 1
                        fv[d]["g_kept"] += len(kg)
                        fv[d]["g_ok"] += all(Cte[k, j] == b for j, b in kg)
                    if ops.forces({j: b for j, b in ki}, pred):
                        fv[d]["i_cov"] += 1
                        fv[d]["i_kept"] += len(ki)
                        fv[d]["i_ok"] += all(Cte[k, j] == b for j, b in ki)
        n = agg["n"]
        qoc = float(np.mean(agg["q_ord_cal"]))
        quc = float(np.mean(agg["q_uar_cal"]))
        escapes = dict(
            n=n,
            oracle_selection=dict(
                p_ord_all_true=agg["ord_all"] / n,
                p_oracle_any_all_true=agg["oracle_any"] / n,
                oracle_gain=(agg["oracle_any"] - agg["ord_all"]) / n,
                oracle_max_supfrac_mean=float(np.mean(agg["oracle_max_supfrac"]))),
            uar_vs_ord=dict(
                frac_identical=agg["uar_eq_ord"] / n,
                q_ord_cal=qoc, q_uar_cal=quc, mean_q_gain_cal=quc - qoc,
                q_ord_raw=float(np.mean(agg["q_ord_raw"])),
                q_uar_raw=float(np.mean(agg["q_uar_raw"])),
                coverage_cal_at_005=float(np.mean(
                    [q >= 0.05 for q in agg["q_uar_cal"]])),
                len_ord=float(np.mean(agg["len_ord"])),
                len_uar=float(np.mean(agg["len_uar"])),
                ord_minconf_lt08=agg["ord_minconf_lt08"] / n,
                ord_minconf_lt05=agg["ord_minconf_lt05"] / n),
            single_clause=dict(
                frac_len1=agg["len1"] / n,
                p_all_given_len1=agg["len1_all"] / agg["len1"] if agg["len1"] else None,
                n_len1=agg["len1"]),
            polarity_survivors=dict(
                frac_neg_only=agg["neg_only"] / n, n_neg_only=agg["neg_only"],
                p_all_given_neg_only=agg["neg_only_all"] / agg["neg_only"] if agg["neg_only"] else None,
                p_all_given_any_positive=agg["any_pos_all"] / agg["any_pos"] if agg["any_pos"] else None,
                n_any_positive=agg["any_pos"]),
            trusted_only={},
            trusted_only_full_vector={},
        )
        for d in deltas:
            t = trust[d]
            row = {}
            for rule, pre in (("global_concept_accuracy", "g"),
                              ("per_instance_calibrated", "i")):
                cov = t[f"{pre}_cov"]
                ok = t[f"{pre}_ok"]
                row[rule] = dict(
                    coverage=cov / n,
                    mean_kept=t[f"{pre}_kept"] / cov if cov else None,
                    p_kept_true_given_covered=ok / cov if cov else None,
                    p_kept_true_ci=list(wilson(ok / cov, cov)) if cov else None,
                    n_covered=cov)
            escapes["trusted_only"][str(d)] = row
            t = fv[d]
            row = {}
            for rule, pre in (("global_concept_accuracy", "g"),
                              ("per_instance_calibrated", "i")):
                cov, ok, nn = t[f"{pre}_cov"], t[f"{pre}_ok"], t["n"]
                row[rule] = dict(
                    coverage=cov / nn,
                    mean_kept=t[f"{pre}_kept"] / cov if cov else None,
                    p_kept_true_given_covered=ok / cov if cov else None,
                    p_kept_true_ci=list(wilson(ok / cov, cov)) if cov else None,
                    n_covered=cov)
            escapes["trusted_only_full_vector"][str(d)] = row
        payload["stage3_escapes"] = escapes
        o = escapes["oracle_selection"]
        print(f"  oracle gain {o['oracle_gain']:.4f} "
              f"(P(any all-true) {o['p_oracle_any_all_true']:.3f} vs ORD "
              f"{o['p_ord_all_true']:.3f})", flush=True)
        u = escapes["uar_vs_ord"]
        print(f"  UAR==ORD on {u['frac_identical']:.3f}; q gain cal "
              f"{u['mean_q_gain_cal']:.4f}; cov@0.05 "
              f"{u['coverage_cal_at_005']:.3f}", flush=True)
        s = escapes["single_clause"]
        v = escapes["polarity_survivors"]
        print(f"  len-1 reasons {s['frac_len1']:.3f} (P(all|len1) "
              f"{s['p_all_given_len1']}); neg-only {v['frac_neg_only']:.3f} "
              f"(P(all) {v['p_all_given_neg_only']}) vs any-pos "
              f"({v['p_all_given_any_positive']})", flush=True)
        for d in deltas:
            g = escapes["trusted_only"][str(d)]["global_concept_accuracy"]
            i = escapes["trusted_only"][str(d)]["per_instance_calibrated"]
            print(f"  trust d={d} (reason-scoped): global cov {g['coverage']:.3f} "
                  f"P(kept-true|cov) {g['p_kept_true_given_covered']}; "
                  f"per-inst cov {i['coverage']:.3f} "
                  f"P(kept-true|cov) {i['p_kept_true_given_covered']}", flush=True)
        for d in deltas:
            g = escapes["trusted_only_full_vector"][str(d)]["global_concept_accuracy"]
            i = escapes["trusted_only_full_vector"][str(d)]["per_instance_calibrated"]
            print(f"  trust d={d} (full-vector): global cov {g['coverage']:.3f} "
                  f"kept {g['mean_kept']} P(kept-true|cov) {g['p_kept_true_given_covered']}; "
                  f"per-inst cov {i['coverage']:.3f} kept {i['mean_kept']} "
                  f"P(kept-true|cov) {i['p_kept_true_given_covered']}", flush=True)
        save()

    # ------------------------------------------------------------- stage 4
    if 4 in stages:
        print("stage 4: certified-concepts retrain (CUB certified_only "
              "protocol) + instance-drop arm ...", flush=True)
        cert = {}
        # arm A: restrict tree vocabulary to concepts whose readers pass a
        # validation-accuracy certification threshold (CUB analogue =
        # reader-certified concepts; CUB grounding controls have no BDD
        # equivalent, so val accuracy is the instrument)
        for thr in (0.75, 0.8, 0.85):
            feats = [j for j in range(N_CONCEPTS) if reader_va[j] >= thr]
            arm = dict(n_concepts=len(feats), per_action={})
            if len(feats) < 3:
                arm["skipped"] = True
                cert[f"concepts_va>={thr}"] = arm
                continue
            for a in ACTIVE_ACTIONS:
                ytr = Ytr[:, a].astype(int)
                if ytr.min() == ytr.max():
                    continue
                Bf = {s: B[s][:, feats] for s in B}
                teacher = LogisticRegression(max_iter=1000, random_state=SEED)
                teacher.fit(Bf["train"].astype(float), ytr)
                tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
                tree.fit(Bf["train"].astype(float), ytr)

                def sub_ops(clf, feats=feats):
                    t = clf.tree_

                    def rec(node):
                        if t.children_left[node] == -1:
                            return {"c": int(np.argmax(t.value[node]))}
                        return {"f": int(feats[t.feature[node]]), "t": 0.5,
                                "l": rec(t.children_left[node]),
                                "r": rec(t.children_right[node])}
                    return TreeOps(dict(classes=[0, 1], n_features=N_CONCEPTS,
                                        root=rec(0),
                                        feature_names=CONCEPT_NAMES))

                ops = sub_ops(tree)
                rows = reasons_on(ops, B["test"], Cte, a, range(len(Xte)))
                at = np.array([r["all_true"] for r in rows])
                yte = Yte[:, a].astype(int)
                sp = np.array([ops.predict_one(x) for x in B["test"].astype(float)])
                arm["per_action"][ACTION_NAMES[a]] = dict(
                    tree_acc=float((sp == yte).mean()),
                    p_all=float(at.mean()) if len(at) else None,
                    p_all_ci=list(wilson(at.mean(), len(at))) if len(at) else None,
                    n=len(rows),
                    reason_len_mean=float(np.mean([len(r["lits"]) for r in rows]))
                    if rows else None)
            tot = sum(v["n"] for v in arm["per_action"].values()
                      if v["n"]) or 1
            arm["p_all_pooled"] = sum(
                v["p_all"] * v["n"] for v in arm["per_action"].values()
                if v["n"]) / tot
            cert[f"concepts_va>={thr}"] = arm
            print(f"  concepts va>={thr}: n_concepts {len(feats)} pooled "
                  f"P(all) {arm['p_all_pooled']:.3f}", flush=True)
        # arm B: drop train instances whose positive certificate has an
        # unsupported premise (negatives kept, so labels stay balanced)
        inst = dict(per_action={})
        for a in ACTIVE_ACTIONS:
            ytr = Ytr[:, a].astype(int)
            _, _, ops0 = fitted[a]
            train_rows = reasons_on(ops0, B["train"], Ctr, a, range(len(Xtr)))
            mask = np.ones(len(Xtr), dtype=bool)
            for r in train_rows:
                if not r["all_true"]:
                    mask[r["idx"]] = False
            if mask.sum() < 50 or ytr[mask].min() == ytr[mask].max():
                inst["per_action"][ACTION_NAMES[a]] = dict(
                    skipped=True, train_kept=float(mask.mean()))
                continue
            teacher, tree, ops = fit_teacher_tree(B["train"][mask], ytr[mask])
            yte = Yte[:, a].astype(int)
            tp = teacher.predict(B["test"].astype(float))
            sp = np.array([ops.predict_one(x) for x in B["test"].astype(float)])
            rows = reasons_on(ops, B["test"], Cte, a, range(len(Xte)))
            at = np.array([r["all_true"] for r in rows])
            inst["per_action"][ACTION_NAMES[a]] = dict(
                train_kept=float(mask.mean()),
                teacher_acc=float((tp == yte).mean()),
                tree_acc=float((sp == yte).mean()),
                p_all=float(at.mean()) if len(at) else None,
                p_all_ci=list(wilson(at.mean(), len(at))) if len(at) else None,
                n=len(rows),
                reason_len_mean=float(np.mean([len(r["lits"]) for r in rows]))
                if rows else None)
            print(f"  drop-unsupported {ACTION_NAMES[a]}: kept {mask.mean():.3f}"
                  f" P(all) {inst['per_action'][ACTION_NAMES[a]]['p_all']}"
                  f" n={len(rows)}", flush=True)
        cert["instance_drop"] = inst
        payload["stage4_certified_only"] = cert
        save()

    # ------------------------------------------------------------- stage 5
    if 5 in stages:
        print("stage 5: stronger (MLP) readers ...", flush=True)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        Xm = {s: torch.from_numpy(data[s][1]).to(dev) for s in data}
        Bm = {s: np.zeros((len(data[s][1]), N_CONCEPTS), dtype=bool)
              for s in data}
        reader_mlp_va = []
        for j in range(N_CONCEPTS):
            y = Ctr[:, j].astype(np.float32)
            if y.min() == y.max():
                for s in data:
                    Bm[s][:, j] = y[0] > 0.5
                reader_mlp_va.append(1.0)
                continue
            torch.manual_seed(SEED + j)
            net = torch.nn.Sequential(
                torch.nn.Linear(2048, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 1))
            net.to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            yv = torch.from_numpy(y).to(dev)
            for ep in range(20):
                pm = net(Xm["train"]).ravel()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pm, yv)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for s in data:
                    p = torch.sigmoid(net(Xm[s]).ravel()).cpu().numpy()
                    Bm[s][:, j] = p >= 0.5
            reader_mlp_va.append(float((Bm["val"][:, j] == Cva[:, j]).mean()))
            print(f"    reader {CONCEPT_NAMES[j]}: val {reader_mlp_va[-1]:.3f}",
                  flush=True)
        print(f"  MLP readers: valid acc mean {np.mean(reader_mlp_va):.3f} "
              f"(logistic {np.mean(reader_va):.3f})", flush=True)
        mlp = dict(reader_valid_acc=[float(x) for x in reader_mlp_va],
                   reader_valid_acc_mean=float(np.mean(reader_mlp_va)),
                   hidden=256, epochs=20, per_action={})
        for a in ACTIVE_ACTIONS:
            ytr = Ytr[:, a].astype(int)
            teacher, tree, ops = fit_teacher_tree(Bm["train"], ytr)
            yte = Yte[:, a].astype(int)
            tp = teacher.predict(Bm["test"].astype(float))
            sp = np.array([ops.predict_one(x) for x in Bm["test"].astype(float)])
            rows = reasons_on(ops, Bm["test"], Cte, a, range(len(Xte)))
            at = np.array([r["all_true"] for r in rows])
            mlp["per_action"][ACTION_NAMES[a]] = dict(
                teacher_acc=float((tp == yte).mean()),
                tree_acc=float((sp == yte).mean()),
                p_all=float(at.mean()) if len(at) else None,
                p_all_ci=list(wilson(at.mean(), len(at))) if len(at) else None,
                n=len(rows),
                reason_len_mean=float(np.mean([len(r["lits"]) for r in rows]))
                if rows else None)
            print(f"  {ACTION_NAMES[a]}: P(all) {mlp['per_action'][ACTION_NAMES[a]]['p_all']}"
                  f" n={len(rows)}", flush=True)
        tot = sum(v["n"] for v in mlp["per_action"].values() if v["n"]) or 1
        mlp["p_all_pooled"] = sum(v["p_all"] * v["n"]
                                  for v in mlp["per_action"].values() if v["n"]) / tot
        mlp["n_pooled"] = tot
        print(f"  MLP pooled P(all) {mlp['p_all_pooled']:.3f} "
              f"(logistic baseline {payload.get('stage1_reproduce', {}).get('pooled', {}).get('p')})",
              flush=True)
        payload["stage5_stronger_reader"] = mlp
        save()

    # ------------------------------------------------------------- stage 6
    if 6 in stages:
        print("stage 6: ladder rungs ...", flush=True)
        ladder = {}
        for a in ACTIVE_ACTIONS:
            ytr, yte = Ytr[:, a].astype(int), Yte[:, a].astype(int)
            direct = LogisticRegression(max_iter=1000, random_state=SEED)
            direct.fit(Xtr, ytr)
            t_gt, tr_gt, ops_gt = fit_teacher_tree(Ctr.astype(float), ytr)
            sp_gt = np.array([ops_gt.predict_one(x) for x in Cte.astype(float)])
            row = dict(
                direct_probe_acc=float(direct.score(Xte, yte)),
                deployed_teacher_acc=payload.get("stage1_reproduce", {})
                .get("per_action", {}).get(ACTION_NAMES[a], {}).get("teacher_acc"),
                deployed_tree_acc=payload.get("stage1_reproduce", {})
                .get("per_action", {}).get(ACTION_NAMES[a], {}).get("tree_acc"),
                gtbits_teacher_acc=float(t_gt.score(Cte.astype(float), yte)),
                gtbits_tree_acc=float((sp_gt == yte).mean()),
                majority=max(ytr.mean(), 1 - ytr.mean()),
            )
            ladder[ACTION_NAMES[a]] = row
            print(f"  {ACTION_NAMES[a]}: direct {row['direct_probe_acc']:.3f} "
                  f"deployed {row['deployed_teacher_acc']} "
                  f"GT-teacher {row['gtbits_teacher_acc']:.3f} "
                  f"GT-tree {row['gtbits_tree_acc']:.3f} "
                  f"(maj {row['majority']:.3f})", flush=True)
        keys = ("direct_probe_acc", "gtbits_teacher_acc", "gtbits_tree_acc",
                "majority")
        ladder["mean_over_actions"] = {
            k: float(np.mean([ladder[ACTION_NAMES[a]][k] for a in ACTIVE_ACTIONS
                              if ladder[ACTION_NAMES[a]][k] is not None]))
            for k in keys}
        payload["stage6_ladder"] = ladder
        save()

    # ------------------------------------------------------------- stage 7
    if 7 in stages:
        print("stage 7: label-noise positive control ...", flush=True)
        lens = np.array([len(r["lits"]) for r in rows_all])
        # iid-flip reference: P(noise GT agrees with literal) per literal is
        # (1-eps) if supported else eps, multiplied within a reason
        noise = {}
        for eps in (0.05, 0.1, 0.2, 0.3):
            obs = []
            for sd in range(5):
                r2 = np.random.RandomState(1000 + sd)
                flip = r2.rand(*Cte.shape) < eps
                Cn = Cte ^ flip
                obs.append(float(np.mean([
                    all(Cn[r["idx"], f] == b for f, b in r["lits"])
                    for r in rows_all])))
            analytic = float(np.mean([
                np.prod([(1 - eps) if s else eps for s in r["sup"]])
                for r in rows_all]))
            noise[str(eps)] = dict(
                observed_mean=float(np.mean(obs)),
                observed_sd=float(np.std(obs)),
                analytic_iid=analytic)
            print(f"  eps={eps}: P(all) {np.mean(obs):.3f} +- {np.std(obs):.3f} "
                  f"(analytic {analytic:.3f})", flush=True)
        noise["baseline_p_all"] = payload.get("stage1_reproduce", {}).get(
            "pooled", {}).get("p")
        payload["stage7_label_noise"] = noise
        save()

    print(f"[done] {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
