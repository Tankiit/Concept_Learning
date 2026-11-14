"""BDD-OIA (rsbench) methods comparison — other explanation methods and
other model families under the same premise-support audit.

Part 1 — explanation methods on the FROZEN sequential pipeline (logistic
readers -> per-action logistic teacher + depth-4 CART, identical to
rsbench_bdd_experiments.py stage 1), same certified-positive test
instances for every method:
  ours      path-restricted backward-elimination minimal sufficient reason
            (exact certification, the committed machinery);
  erasure   behavior-explainer-style deletion: greedily erase bits to the
            absent-baseline 0 (order = descending |teacher coef|) while the
            tree prediction is preserved; explanation = kept literals;
  topk      global saliency straw man: top-3 |teacher coef| bits per action,
            asserted with the instance's values;
  random    k=3 random bits per instance (seeded), asserted values.
Per method: mean size, premise support P(all asserted literals GT-true)
with Wilson CI, exact sufficiency on the tree (forces), teacher-flip
fidelity (P(teacher prediction changes when the selected bits flip)).

Part 2 — model families (rsbench taxonomy, their architectures on their
2048-d tensors, seed 42): the readers/conceptizer are held fixed (their
BOIAConceptizer is a linear 2048->21 probe = our logistic reader modulo
temperature); families differ in where the task head's training signal
comes from:
  joint        their boiann/BOIAMLP 2048-1024-512-256-64-4 sigmoid, BCE
               per label — no concept bottleneck (post-hoc concept student);
  independent  their boiacbm: linear head trained on GT concept bits,
               deployed on predicted bits (classic CBM / RS setting);
  sequential   our committed pipeline: head trained on predicted bits;
  sequential-T the committed audit tree trained on true labels (Setting T).
Every family gets the same Boolean student (CART depth 4 over the 21
reader bits, distilled from the family's own train predictions) and the
same audit: per-action acc, exact match, tree agreement, minimal reasons,
premise support.

Output: results/bdd_oia_methods.json (rewritten after each part).

Env: ltn_pyxai_starter/.venv. Data: tmp/datasets/rsbench/bdd2048 (--data-root).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
ACTIVE_ACTIONS = [0, 1, 2, 3]


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
    return X, C, Y


def tree_ops_from(clf):
    t = clf.tree_

    def rec(node):
        if t.children_left[node] == -1:
            return {"c": int(np.argmax(t.value[node]))}
        return {"f": int(t.feature[node]), "t": 0.5,
                "l": rec(t.children_left[node]), "r": rec(t.children_right[node])}

    return TreeOps(dict(classes=[0, 1], n_features=N_CONCEPTS, root=rec(0),
                        feature_names=CONCEPT_NAMES))


def lit_all_true(lits, k, C):
    return all(bool(C[k, f] == b) for f, b in lits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(GV / "tmp" / "datasets" / "rsbench" / "bdd2048"))
    ap.add_argument("--parts", default="1,2")
    ap.add_argument("--out", default=str(GV / "results" / "bdd_oia_methods.json"))
    args = ap.parse_args()
    parts = {int(p) for p in args.parts.split(",")}
    root = Path(args.data_root)
    rng = np.random.RandomState(SEED)
    out_path = Path(args.out)
    payload = dict(
        benchmark="rsbench BDD-OIA (bdd2048 preprocessed)", seed=SEED,
        protocol=dict(
            data="their Zenodo preprocessed tensors, official splits",
            conceptizer="logistic per concept (= their BOIAConceptizer "
                        "Linear(2048,21) modulo temperature; argmax-identical)",
            student="CART depth 4 over the 21 reader bits",
        ),
    )

    def save():
        out_path.write_text(json.dumps(payload, indent=1))

    t0 = time.time()
    print("loading tensors ...", flush=True)
    Xtr, Ctr, Ytr = load_split(root, "train")
    Xva, Cva = load_split(root, "val")[:2]
    Xte, Cte, Yte = load_split(root, "test")
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

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

    P = {s: probs(x) for s, x in (("train", Xtr), ("val", Xva), ("test", Xte))}
    B = {s: (P[s] >= 0.5) for s in P}
    reader_va = float(np.mean([(B["val"][:, j] == Cva[:, j]).mean()
                               for j in range(N_CONCEPTS)]))
    print(f"  readers valid acc mean {reader_va:.3f}", flush=True)

    # frozen sequential pipeline (== experiments stage 1)
    teacher_seq, trees, ops_all = {}, {}, {}
    for a in ACTIVE_ACTIONS:
        ytr = Ytr[:, a].astype(int)
        t = LogisticRegression(max_iter=1000, random_state=SEED)
        t.fit(B["train"].astype(float), ytr)
        teacher_seq[a] = t
        tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        tree.fit(B["train"].astype(float), ytr)
        trees[a] = tree
        ops_all[a] = tree_ops_from(tree)

    # ------------------------------------------------------------- part 1
    if 1 in parts:
        print("part 1: explanation methods on the frozen pipeline ...",
              flush=True)
        methods = ("ours", "erasure", "topk", "random")
        agg = {m: dict(n=0, size=[], all_true=[], forces=[], flip=[],
                       n_pos_lit=0, n_lit=0) for m in methods}
        per_action = {a: {m: dict(n=0, ok=0) for m in methods}
                      for a in ACTIVE_ACTIONS}
        for a in ACTIVE_ACTIONS:
            ops, teacher = ops_all[a], teacher_seq[a]
            coef = teacher.coef_[0]
            order = np.argsort(-np.abs(coef))
            topk_bits = list(order[:3])
            base_pred = teacher.predict(B["test"].astype(float))
            for k in range(len(Xte)):
                x = B["test"][k].astype(float)
                if ops.predict_one(x) != 1:
                    continue
                # ours: minimal sufficient reason
                ours = [(int(f), int(b)) for f, b in ops.minimal_reason(x, 1)]
                # erasure: greedy to baseline 0 in |coef| order
                cur = B["test"][k].copy()
                kept = []
                for j in order:
                    if cur[j] == 0:
                        continue
                    trial = cur.copy()
                    trial[j] = 0
                    if ops.predict_one(trial.astype(float)) == 1:
                        cur = trial
                    else:
                        kept.append((int(j), 1))
                # bits already 0 are never erased-to-something: only kept
                # positive bits enter the erasure explanation (deletion
                # methods assert presence, not absence)
                # topk global
                topk = [(int(j), int(B["test"][k, j])) for j in topk_bits]
                rnd_idx = rng.choice(N_CONCEPTS, 3, replace=False)
                randk = [(int(j), int(B["test"][k, j])) for j in rnd_idx]
                for name, lits in (("ours", ours), ("erasure", kept),
                                   ("topk", topk), ("random", randk)):
                    xflip = B["test"][k].copy()
                    for f, b in lits:
                        xflip[f] = 1 - b
                    flipped = teacher.predict(
                        xflip.astype(float).reshape(1, -1))[0]
                    m = agg[name]
                    m["n"] += 1
                    m["size"].append(len(lits))
                    m["all_true"].append(lit_all_true(lits, k, Cte))
                    m["forces"].append(bool(ops.forces(
                        {f: b for f, b in lits}, 1)) if lits else False)
                    m["flip"].append(bool(flipped != base_pred[k]))
                    m["n_lit"] += len(lits)
                    m["n_pos_lit"] += sum(b for _, b in lits)
                    per_action[a][name]["n"] += 1
                    per_action[a][name]["ok"] += int(lit_all_true(lits, k, Cte))
        out1 = dict(reader_valid_acc=reader_va, methods={})
        for name, m in agg.items():
            at = np.array(m["all_true"])
            p, (lo, hi) = float(at.mean()), wilson(at.mean(), len(at))
            out1["methods"][name] = dict(
                n=m["n"], size_mean=float(np.mean(m["size"])),
                p_all=p, lo=lo, hi=hi,
                frac_positive_literals=(m["n_pos_lit"] / m["n_lit"]
                                        if m["n_lit"] else None),
                forces_rate=float(np.mean(m["forces"])),
                teacher_flip_rate=float(np.mean(m["flip"])))
            print(f"  {name:<8} size {np.mean(m['size']):.2f} "
                  f"P(all) {p:.3f} [{lo:.3f},{hi:.3f}] "
                  f"pos-frac {(m['n_pos_lit'] / m['n_lit'] if m['n_lit'] else float('nan')):.3f} "
                  f"forces {np.mean(m['forces']):.3f} "
                  f"teacher-flip {np.mean(m['flip']):.3f}", flush=True)
        out1["per_action_p_all"] = {
            ACTION_NAMES[a]: {m: (v["ok"] / v["n"] if v["n"] else None)
                              for m, v in per_action[a].items()}
            for a in ACTIVE_ACTIONS}
        payload["part1_explanation_methods"] = out1
        save()

    # ------------------------------------------------------------- part 2
    if 2 in parts:
        print("part 2: model families (rsbench taxonomy) ...", flush=True)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"

        # joint: their BOIAMLP architecture on the 2048-d tensors
        torch.manual_seed(SEED)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 4), torch.nn.Sigmoid()).to(dev)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        Xt = torch.from_numpy(Xtr).to(dev)
        Yt = torch.from_numpy(Ytr[:, ACTIVE_ACTIONS].astype(np.float32)).to(dev)
        Xe = torch.from_numpy(Xte).to(dev)
        for ep in range(30):
            pm = mlp(Xt)
            loss = torch.nn.functional.binary_cross_entropy(pm, Yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            joint_te = (mlp(Xe).cpu().numpy() >= 0.5)
            joint_tr = (mlp(Xt).cpu().numpy() >= 0.5)
        print(f"  joint NN trained (final BCE {loss.item():.4f})", flush=True)

        # independent head: linear on GT concept bits (their boiacbm head),
        # deployed on predicted bits
        teacher_gt, ind_tr, ind_te = {}, {}, {}
        for i, a in enumerate(ACTIVE_ACTIONS):
            t = LogisticRegression(max_iter=1000, random_state=SEED)
            t.fit(Ctr.astype(float), Ytr[:, a].astype(int))
            teacher_gt[a] = t
            ind_tr[a] = t.predict(B["train"].astype(float))
            ind_te[a] = t.predict(B["test"].astype(float))

        seq_tr = {a: teacher_seq[a].predict(B["train"].astype(float))
                  for a in ACTIVE_ACTIONS}
        seq_te = {a: teacher_seq[a].predict(B["test"].astype(float))
                  for a in ACTIVE_ACTIONS}

        families = {}
        fam_preds = {
            "joint_mlp": (joint_tr, joint_te, "x"),
            "cbm_independent": (ind_tr, ind_te, "bits"),
            "sequential": (seq_tr, seq_te, "bits"),
        }
        for fam, (ptr, pte, _) in fam_preds.items():
            row = dict(per_action={}, concept_acc=reader_va)
            reasons_all = []
            eqs = []
            for i, a in enumerate(ACTIVE_ACTIONS):
                yte = Yte[:, a].astype(int)
                fam_te = pte[:, i] if fam == "joint_mlp" else pte[a]
                fam_tr = ptr[:, i] if fam == "joint_mlp" else ptr[a]
                eqs.append(fam_te == yte)
                # student: CART depth-4 over reader bits, distilled from the
                # family's own train predictions
                tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
                tree.fit(B["train"].astype(float), fam_tr)
                ops = tree_ops_from(tree)
                sp = np.array([ops.predict_one(x)
                               for x in B["test"].astype(float)])
                acc = float((fam_te == yte).mean())
                agr = float((sp == fam_te).mean())
                n_pos = 0
                ok = 0
                lens = []
                for k in range(len(Xte)):
                    if sp[k] != 1:
                        continue
                    r = [(int(f), int(b))
                         for f, b in ops.minimal_reason(B["test"][k].astype(float), 1)]
                    n_pos += 1
                    lens.append(len(r))
                    t_ = lit_all_true(r, k, Cte)
                    ok += int(t_)
                    reasons_all.append(t_)
                p, (lo, hi) = (ok / n_pos if n_pos else None), wilson(
                    ok / n_pos, n_pos) if n_pos else (None, None)
                row["per_action"][ACTION_NAMES[a]] = dict(
                    fam_acc=acc, tree_acc=float((sp == yte).mean()),
                    agreement=agr, p_all=p, lo=lo, hi=hi, n=n_pos,
                    reason_len_mean=float(np.mean(lens)) if lens else None)
            at = np.array(reasons_all)
            pp, (lo, hi) = float(at.mean()), wilson(at.mean(), len(at))
            exact = float(np.stack(eqs, 1).all(1).mean())
            row.update(p_all_pooled=pp, lo=lo, hi=hi, n=len(at),
                       exact_match_all4=exact,
                       mean_acc=float(np.mean(
                           [row["per_action"][ACTION_NAMES[a]]["fam_acc"]
                            for a in ACTIVE_ACTIONS])))
            families[fam] = row
            print(f"  {fam:<15} mean acc {row['mean_acc']:.3f} "
                  f"exact4 {row['exact_match_all4']:.3f} "
                  f"pooled P(all) {pp:.3f} [{lo:.3f},{hi:.3f}] n={len(at)}",
                  flush=True)

        # sequential-T: the committed audit setting (tree on true labels)
        row = dict(per_action={}, concept_acc=reader_va,
                   note="committed audit pipeline (Setting T)")
        reasons_all = []
        exact_t = []
        for a in ACTIVE_ACTIONS:
            ops = ops_all[a]
            yte = Yte[:, a].astype(int)
            sp = np.array([ops.predict_one(x)
                           for x in B["test"].astype(float)])
            seq_te_a = seq_te[a]
            exact_t.append(seq_te_a == yte)
            n_pos = ok = 0
            lens = []
            for k in range(len(Xte)):
                if sp[k] != 1:
                    continue
                r = [(int(f), int(b))
                     for f, b in ops.minimal_reason(B["test"][k].astype(float), 1)]
                n_pos += 1
                lens.append(len(r))
                ok += int(lit_all_true(r, k, Cte))
                reasons_all.append(lit_all_true(r, k, Cte))
            p, (lo, hi) = ok / n_pos, wilson(ok / n_pos, n_pos)
            row["per_action"][ACTION_NAMES[a]] = dict(
                fam_acc=float((seq_te_a == yte).mean()),
                tree_acc=float((sp == yte).mean()),
                agreement=float((sp == seq_te_a).mean()),
                p_all=p, lo=lo, hi=hi, n=n_pos,
                reason_len_mean=float(np.mean(lens)) if lens else None)
        at = np.array(reasons_all)
        pp, (lo, hi) = float(at.mean()), wilson(at.mean(), len(at))
        row.update(p_all_pooled=pp, lo=lo, hi=hi, n=len(at),
                   exact_match_all4=float(np.mean(np.stack(exact_t, 1).all(1))),
                   mean_acc=float(np.mean(
                       [row["per_action"][ACTION_NAMES[a]]["fam_acc"]
                        for a in ACTIVE_ACTIONS])))
        families["sequential_T_committed"] = row
        print(f"  {'sequential_T':<15} mean acc {row['mean_acc']:.3f} "
              f"exact4 {row['exact_match_all4']:.3f} pooled P(all) {pp:.3f} "
              f"[{lo:.3f},{hi:.3f}] n={len(at)}", flush=True)

        payload["part2_families"] = families
        save()

    print(f"[done] {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
