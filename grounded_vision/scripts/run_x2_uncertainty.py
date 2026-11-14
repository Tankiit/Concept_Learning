"""
Script: run_x2_uncertainty.py
=============================
X2 (2026-11-08): ordinary vs uncertainty-aware sufficient reasons on
FROZEN models, per the clarified protocol:

  - visual backbone + concept readers (logistic probes on pooled region
    features) are neural, produce per-concept scores s_j in [0,1];
  - the final classifier is an ORDINARY hard-routed decision tree
    (sklearn CART) fitted on the readers' thresholded bits -- either on
    the labels (Setting T) or distilled from a frozen linear head
    (Setting D, Design B);
  - everything is FROZEN while the two explanation methods are compared
    on the SAME tree (no joint training, no soft routing; the tree is
    already in the form formal explainers operate on).

Completion model (per-image reader uncertainty): a completion is a bit
vector b' with b'_j ~ Bernoulli(p_j).  Two arms:
  raw   p_j = s_j (the reader sigmoid; NOTE readers are trained with
        pos_weight, which distorts calibration toward 0.5)
  cal   p_j = isotonic_j(s_j) fitted on the validation carve-out
        against the GT attribute bit (exploratory robustness arm,
        added after the raw-arm run; NOT pre-registered)
Structural features (has_part, unobserved parts) are deterministic.

For a sufficient reason R (a set of path literals, which by exact
sufficiency forces class c whenever satisfied):

  q(R) = P(b' satisfies R) = prod_{(j,v) in R} [p_j if v=1 else 1-p_j]

is simultaneously (i) the probability the explanation's literals are
true under the completion model and (ii) the probability a completion
both satisfies R and is classified c (sufficiency is exact).  Methods
compared on the same frozen tree, same path-literal pool:

  ORD  ordinary minimum-size sufficient reason (native backward
       elimination -- the committed machinery)
  UAR  uncertainty-aware reason: the sufficient subset of the path
       literals MAXIMISING q(R) (exhaustive over <=2^12 path subsets,
       exact), tie-break shorter.

PRE-REGISTERED PREDICTIONS (before the raw-arm run):
  X2-P1 at seed 42, K=4, setting T: >= 50% of instances have an ORD
       reason containing >= 1 literal with reader confidence < 0.8.
  X2-P2 UAR > ORD in mean q / coverage at delta=0.05 somewhere, with
       mean-q gain >= 0.05 somewhere.
  X2-P3 mean |R_UAR| >= mean |R_ORD|.
  X2-P4 UAR coverage at delta=0.05 stays < 0.90 at K=4 setting T
       (measured limit).
  X2-P5 conditional fidelity to the teacher differs by < 0.10 between
       methods on shared patterns.

RAW-ARM OUTCOME (2026-11-08, seed 42; committed x2 JSON): UAR == ORD
identically on all 722 instances -- diag_x2_uniqueness.py shows the
minimal sufficient subset of the path literals is UNIQUE per instance
(118/118 at T K=4) and every ORD literal is leave-one-out necessary,
so the max-q sufficient set is the minimal set and selection has no
room.  X2-P2 is falsified in the strongest form (gain identically 0);
X2-P1 holds at 100%; X2-P4 holds at coverage 0.0.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def leaf_bounds(clf):
    """Per-leaf (lower_exclusive, upper_inclusive) bounds per feature."""
    tree = clf.tree_
    leaves = []

    def recurse(node, lo, hi):
        if tree.children_left[node] == -1 or tree.feature[node] < 0:
            counts = tree.value[node][0]
            leaves.append({
                "pred": clf.classes_[int(np.argmax(counts))],
                "lo": dict(lo), "hi": dict(hi),
            })
            return
        f, t = int(tree.feature[node]), float(tree.threshold[node])
        lo2 = dict(lo); lo2[f] = max(lo.get(f, -np.inf), t)
        hi2 = dict(hi); hi2[f] = min(hi.get(f, np.inf), t)
        recurse(tree.children_left[node], lo, hi2)
        recurse(tree.children_right[node], lo2, hi)

    recurse(0, {}, {})
    return leaves


def literal_contradicts_leaf(fidx, is_left, thr, leaf):
    """Mirror of the verifier's interval logic for one path literal vs leaf."""
    if is_left:  # feature <= thr
        if fidx in leaf["lo"]:
            return thr <= leaf["lo"][fidx]
        return False
    else:       # feature > thr
        if fidx in leaf["hi"]:
            return thr >= leaf["hi"][fidx]
        return False


def uar_from_path(path_lits, path_leaf_pred, leaves, confs):
    """Exact max-q sufficient subset of the path literals.

    Sufficient(S) iff every leaf with pred != target is blocked by some
    literal in S (contradicts its path). The reached leaf is consistent
    with any subset of its own path, so consistent_count > 0 always.
    The full path blocks every other leaf, so a solution exists whenever
    the ordinary reason does.
    """
    neg_masks = []
    for lf in leaves:
        if lf["pred"] == path_leaf_pred:
            continue
        mask = 0
        for i, (fidx, is_left, thr) in enumerate(path_lits):
            if literal_contradicts_leaf(fidx, is_left, thr, lf):
                mask |= (1 << i)
        if mask == 0:
            return None  # a wrong-class leaf is unblockable from the path
        neg_masks.append(mask)
    m = len(path_lits)
    logc = [-math.log(max(c, 1e-12)) for c in confs]
    best = None  # (q, -length, -subset_bits); ties -> shorter, then larger idx set
    for s in range(1, 1 << m):
        if any((nm & s) == 0 for nm in neg_masks):
            continue
        q = math.exp(-sum(logc[i] for i in range(m) if s >> i & 1))
        cand = (q, -(bin(s).count("1")), -s)
        if best is None or cand > best:
            best = cand
    if best is None:
        return None
    s = -best[2]
    idxs = [i for i in range(m) if s >> i & 1]
    return {"idxs": idxs, "q": best[0], "length": len(idxs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="4,10")
    ap.add_argument("--class-ids", default=None,
                    help="comma-separated explicit class labels; overrides the "
                         "label<K first-K selection (subset-sensitivity draws)")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--deltas", default="0.05,0.2")
    ap.add_argument("--min-group", type=int, default=5)
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    deltas = [float(d) for d in args.deltas.split(",")]
    class_ids = (set(int(c) for c in args.class_ids.split(","))
                 if args.class_ids else None)

    dtr = np.load(Path(args.cache_dir) / "train_features.npz")
    dte = np.load(Path(args.cache_dir) / "test_features.npz")
    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=42)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=42)
    vocab = ds_train.bound_attributes
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=list(vocab))
    feat_names = compiler.feature_names
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    attr_j_by_feat = {compiler.feature_to_idx[f"{a.part}:{a.attr_name}={a.value}"]: j
                      for j, a in enumerate(vocab)}
    part_feat_idx = [compiler.feature_to_idx[n]
                     for n in feat_names if n.startswith("has_part(")]

    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])

    def pooled(rows, fmaps):
        out = {}
        fm = torch.from_numpy(fmaps.astype(np.float32))
        for i, row in enumerate(rows):
            sid, _m, boxes, vis, _a = row
            regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                    for p in ds_train.part_names if vis.get(p, 0.0) > 0.5 and p in boxes]
            if regs:
                p = pool_region_features(fm[i:i + 1], regs)
                out[int(sid)] = {r.part_name: p[r.id].numpy() for r in regs}
        return out

    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}
    te_order = {int(m[0]): i for i, m in enumerate(te_rows)}
    tr_order = {int(m[0]): i for i, m in enumerate(tr_rows)}
    tr_fm, te_fm = dtr["feature_maps"], dte["feature_maps"]

    for seed in seeds:
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        print(f"==> seed {seed}: pooling train {len(tr_rows)} / test {len(te_rows)}",
              flush=True)
        tr_pooled = pooled(tr_rows, tr_fm)
        te_pooled = pooled(te_rows, te_fm)

        sids = np.array(sorted(tr_pooled.keys()))
        perm = rng.permutation(len(sids))
        n_val = int(args.val_fraction * len(sids))
        val_sids = set(sids[perm[:n_val]].tolist())
        fit_sids = set(sids[perm[n_val:]].tolist())
        reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
        opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
        region_rows = defaultdict(list)
        for sid in fit_sids:
            for part, vec in tr_pooled[sid].items():
                region_rows[part].append((sid, vec))
        region_data = {}
        for part, rows in region_rows.items():
            cols = attr_by_part.get(part)
            if not cols:
                continue
            X = torch.from_numpy(np.stack([v for _, v in rows])).float()
            y = torch.from_numpy(np.array(
                [tr_gt[s][cols] for s, _ in rows], dtype=np.float32))
            pos, neg = y.sum(dim=0), y.shape[0] - y.sum(dim=0)
            pw = (neg / pos.clamp(min=1.0)).clamp(max=50.0)
            region_data[part] = (X, y, pw, cols)
        for _ in range(args.epochs):
            loss = 0.0
            for X, y, pw, cols in region_data.values():
                W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
                b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
                loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"--> seed {seed}: readers trained", flush=True)

        def attr_feat_name(j):
            a = vocab[j]
            return f"{a.part}:{a.attr_name}={a.value}"

        def scores_map(parts):
            d = {}
            for part, vec in parts.items():
                for j in attr_by_part.get(part, []):
                    with torch.no_grad():
                        t = torch.from_numpy(vec.reshape(1, -1)).float()
                        d[attr_feat_name(j)] = float(
                            torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
            return d

        # ---- calibrated completion probabilities (isotonic per attribute,
        #      fit on the validation carve-out against GT attribute bits) ---
        val_scores = defaultdict(list)   # feature_idx -> [(score, gt_bit)]
        for sid in val_sids:
            d = scores_map(tr_pooled[sid])
            for name, sc in d.items():
                fi = compiler.feature_to_idx[name]
                val_scores[fi].append((sc, float(tr_gt[sid][attr_j_by_feat[fi]])))
        isotonic = {}
        for fi, pairs in val_scores.items():
            if len(pairs) < 10:
                continue
            xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
            if ys.min() == ys.max():  # constant target: deterministic
                isotonic[fi] = ("const", float(ys[0]))
            else:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(xs, ys)
                isotonic[fi] = ("ir", ir)

        def cal_prob(fi, s):
            if fi not in isotonic:
                return s
            kind, obj = isotonic[fi]
            return float(obj) if kind == "const" else float(obj.predict([s])[0])

        def vecs(parts):
            """bits_T (has_part=1), bits_D (has_part=0), raw + calibrated scores."""
            d = scores_map(parts)
            sc = np.zeros(len(feat_names), dtype=np.float64)
            sc_cal = np.zeros(len(feat_names), dtype=np.float64)
            for name, p in d.items():
                fi = compiler.feature_to_idx[name]
                sc[fi] = p
                sc_cal[fi] = cal_prob(fi, p)
            bthr = (sc >= 0.5).astype(np.int64)
            bT = bthr.copy()
            for fi in part_feat_idx:
                bT[fi] = 1
            return bT, bthr, sc, sc_cal

        tr_bT, tr_bD, tr_sc, tr_scc = {}, {}, {}, {}
        for s in sorted(fit_sids):
            a, b, c, e = vecs(tr_pooled[s])
            tr_bT[s], tr_bD[s], tr_sc[s], tr_scc[s] = a, b, c, e
        te_bT, te_bD, te_sc, te_scc = {}, {}, {}, {}
        for s in te_pooled.keys():
            a, b, c, e = vecs(te_pooled[s])
            te_bT[s], te_bD[s], te_sc[s], te_scc[s] = a, b, c, e

        out_path = Path(args.out if seed == 42
                        else str(Path(args.out)).replace(".json", f"_seed{seed}.json"))
        blocks = []
        for K in classes:
            sel = class_ids if class_ids is not None else range(int(K))
            tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] in sel]
            te_ids = [s for s in te_pooled.keys() if sid_to_label_te[s] in sel]
            y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
            y_te = np.array([sid_to_label_te[s] for s in te_ids])

            # ---- frozen Setting T: CART on labels over bits_T --------------
            X_tr_T = np.stack([tr_bT[s] for s in tr_ids])
            X_te_T = np.stack([te_bT[s] for s in te_ids])
            dtT = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                         random_state=42).fit(X_tr_T, y_tr)
            headT = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                                        feature_names=feat_names)
            headT.clf, headT.is_fitted = dtT, True
            backT = PyXAIBackend(headT)
            teacherT = LogisticRegression(max_iter=2000, C=1.0,
                                          random_state=42).fit(X_tr_T, y_tr)

            # ---- frozen Setting D: patch head -> distilled CART on bits_D --
            Xp_tr = np.stack([tr_fm[tr_order[s]].astype(np.float32).mean(axis=(1, 2))
                              for s in tr_ids])
            Xp_te = np.stack([te_fm[te_order[s]].astype(np.float32).mean(axis=(1, 2))
                              for s in te_ids])
            patch_head = LogisticRegression(max_iter=3000, C=1.0,
                                            random_state=42).fit(Xp_tr, y_tr)
            X_tr_D = np.stack([tr_bD[s] for s in tr_ids])
            X_te_D = np.stack([te_bD[s] for s in te_ids])
            dtD = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4,
                                         random_state=42).fit(
                                             X_tr_D, patch_head.predict(Xp_tr))
            headD = CertifiableTreeHead(max_depth=4, min_samples_leaf=4,
                                        feature_names=feat_names)
            headD.clf, headD.is_fitted = dtD, True
            backD = PyXAIBackend(headD)
            leavesD = leaf_bounds(dtD)

            for sname, dt, back, X_te, teacher_pred, leaves in (
                ("T_tree_head", dtT, backT, X_te_T, teacherT.predict(X_te_T),
                 leaf_bounds(dtT)),
                ("D_distilled", dtD, backD, X_te_D, patch_head.predict(Xp_te),
                 leavesD),
            ):
                bits_setting = te_bT if sname == "T_tree_head" else te_bD
                tree_pred = dt.predict(X_te)
                n = len(te_ids)
                rows = []
                mc_rng = np.random.RandomState(seed * 1000 + 7)
                for i, s in enumerate(te_ids):
                    v = bits_setting[s]
                    c = int(tree_pred[i])
                    exp = back.explain_sufficient(v, c, sample_id=s)
                    ord_lits = [(int(cc.feature_idx), 1 if cc.operator != "<=" else 0)
                                for cc in exp.conditions]
                    steps = headT.get_decision_path(v) if sname == "T_tree_head" \
                        else headD.get_decision_path(v)
                    path_lits = [(st.feature_idx, st.is_left, st.threshold)
                                 for st in steps]
                    leaf_pred = int(dt.predict(v.reshape(1, -1))[0])

                    rec = {"sid": int(s), "pred": c, "true": int(y_te[i]),
                           "len_ord": len(ord_lits)}
                    stoch = np.where((te_sc[s] > 1e-9) & (te_sc[s] < 1 - 1e-9))[0]
                    rec["n_stoch"] = int(len(stoch))
                    comp_base = np.repeat(v.reshape(1, -1), args.B, axis=0).astype(np.int64)
                    for arm, score_vec in (("raw", te_sc[s]), ("cal", te_scc[s])):
                        sc = score_vec
                        ord_conf = [sc[f] if b == 1 else 1.0 - sc[f]
                                    for f, b in ord_lits]
                        q_ord = float(np.prod([max(cf, 1e-12) for cf in ord_conf]))
                        path_confs = [sc[f] if not is_left else 1.0 - sc[f]
                                      for f, is_left, _t in path_lits]
                        uar = uar_from_path(path_lits, leaf_pred, leaves, path_confs)
                        uar_lits = ([(path_lits[k][0], 0 if path_lits[k][1] else 1)
                                     for k in uar["idxs"]] if uar else None)
                        q_uar = float(uar["q"]) if uar else None
                        draws = mc_rng.random_sample((args.B, len(sc))) < sc
                        comp = comp_base.copy()
                        s_arm = np.where((sc > 1e-9) & (sc < 1 - 1e-9))[0]
                        if len(s_arm):
                            comp[:, s_arm] = draws[:, s_arm].astype(np.int64)
                        for fi in part_feat_idx:
                            comp[:, fi] = 1 if sname == "T_tree_head" else 0
                        comp_pred = dt.predict(comp)
                        rec[f"stability_{arm}"] = round(
                            float(np.mean(comp_pred == c)), 4)
                        rec[f"q_ord_{arm}"] = round(q_ord, 4)
                        rec[f"q_uar_{arm}"] = (round(q_uar, 4)
                                               if q_uar is not None else None)
                        rec[f"min_conf_ord_{arm}"] = round(
                            float(min(ord_conf)), 4) if ord_conf else None
                        rec[f"hamming_{arm}"] = round(float(np.mean(
                            np.sum(comp != v.reshape(1, -1), axis=1))), 2)
                        rec[f"mc_q_ord_{arm}"] = round(float(np.mean(np.all(
                            comp[:, [f for f, _ in ord_lits]] ==
                            np.array([b for _, b in ord_lits]), axis=1))), 4)
                        if uar_lits is not None:
                            rec[f"mc_q_uar_{arm}"] = round(float(np.mean(np.all(
                                comp[:, [f for f, _ in uar_lits]] ==
                                np.array([b for _, b in uar_lits]), axis=1))), 4)
                            set_o, set_u = set(ord_lits), set(uar_lits)
                            rec["jaccard"] = round(
                                len(set_o & set_u) / len(set_o | set_u), 3)
                            rec["len_uar"] = len(uar_lits)
                        else:
                            rec["len_uar"] = None
                    rec["lits_ord"] = [f"{feat_names[f]}{'+' if b else '-'}"
                                       for f, b in ord_lits]
                    rows.append(rec)

                def agg(arm):
                    qs_o = np.array([r[f"q_ord_{arm}"] for r in rows])
                    qs_u = np.array([r[f"q_uar_{arm}"] if r[f"q_uar_{arm}"] is not None
                                     else 0.0 for r in rows])
                    lens_o = np.array([r["len_ord"] for r in rows], dtype=float)
                    lens_u = np.array([r["len_uar"] if r["len_uar"] is not None else 99
                                       for r in rows], dtype=float)
                    cov = {f"ord@{d}": float(np.mean(qs_o >= 1 - d)) for d in deltas}
                    cov.update({f"uar@{d}": float(np.mean(qs_u >= 1 - d))
                                for d in deltas})
                    matched = []
                    for L in range(1, 9):
                        mo, mu = lens_o <= L, lens_u <= L
                        matched.append({
                            "L": L,
                            "cov_len_ord": round(float(np.mean(mo)), 4),
                            "cov_len_uar": round(float(np.mean(mu)), 4),
                            "cov_len_q95_ord": round(float(np.mean(mo & (qs_o >= 0.95))), 4),
                            "cov_len_q95_uar": round(float(np.mean(mu & (qs_u >= 0.95))), 4),
                        })
                    return {
                        "mean_len_ord": round(float(lens_o.mean()), 3),
                        "mean_len_uar": (round(float(lens_u[lens_u < 99].mean()), 3)
                                         if (lens_u < 99).any() else None),
                        "mean_q_ord": round(float(qs_o.mean()), 4),
                        "mean_q_uar": round(float(qs_u.mean()), 4),
                        "frac_minconf_lt08_ord": round(float(np.mean(
                            [r[f"min_conf_ord_{arm}"] < 0.8 for r in rows])), 4),
                        "mean_stability": round(float(np.mean(
                            [r[f"stability_{arm}"] for r in rows])), 4),
                        "mean_hamming": round(float(np.mean(
                            [r[f"hamming_{arm}"] for r in rows])), 2),
                        "coverage": cov, "matched_length": matched,
                    }

                def fidelity(lit_key):
                    pats = {}
                    for r in rows:
                        key = tuple(sorted(r[lit_key])) if r[lit_key] else None
                        if key is None:
                            continue
                        pats.setdefault(key, r["pred"])
                    out = []
                    for key, c in pats.items():
                        fl = [(feat_names.index(n[:-1]), 1 if n.endswith("+") else 0)
                              for n in key]
                        mask = np.all(X_te[:, [f for f, _ in fl]] ==
                                      np.array([b for _, b in fl]), axis=1)
                        n_g = int(mask.sum())
                        if n_g < args.min_group:
                            continue
                        k_g = int((teacher_pred[mask] == c).sum())
                        out.append({"literals": list(key), "n": n_g,
                                    "fidelity": round(k_g / n_g, 4),
                                    "wilson95": wilson(k_g, n_g)})
                    out.sort(key=lambda r: -r["n"])
                    return out

                block = {
                    "setting": sname, "K": K, "seed": seed, "n_test": n,
                    "tree_acc": round(float(np.mean(tree_pred == y_te)), 4),
                    "teacher_agreement": round(
                        float(np.mean(teacher_pred == tree_pred)), 4),
                    "arm_raw": agg("raw"), "arm_cal": agg("cal"),
                    "mean_n_stoch": round(float(np.mean(
                        [r["n_stoch"] for r in rows])), 2),
                    "fidelity_ord": fidelity("lits_ord"),
                    "per_instance": rows,
                }
                blocks.append(block)
                ar, ac = block["arm_raw"], block["arm_cal"]
                print(f"--> seed {seed} {sname} K={K}: "
                      f"RAW q {ar['mean_q_ord']:.3f} cov@.05 {ar['coverage']['ord@0.05']:.3f} "
                      f"stab {ar['mean_stability']:.3f} ham {ar['mean_hamming']:.1f} | "
                      f"CAL q {ac['mean_q_ord']:.3f} cov@.05 {ac['coverage']['ord@0.05']:.3f} "
                      f"stab {ac['mean_stability']:.3f} ham {ac['mean_hamming']:.1f}",
                      flush=True)

        def blk(setting, k):
            return next(b for b in blocks if b["setting"] == setting and b["K"] == k)
        verdicts = None
        # Pre-registered X2 verdicts are defined on the K=4 T block; only computable
        # when K=4 is part of this run (e.g. the task1 driver runs K=20/50/200 only).
        if seed == 42 and any(b["setting"] == "T_tree_head" and b["K"] == 4
                              for b in blocks):
            bt = blk("T_tree_head", 4)
            ar = bt["arm_raw"]
            verdicts = {
                "X2-P1_ord_has_unconfident_lit": bool(
                    ar["frac_minconf_lt08_ord"] >= 0.5),
                "X2-P2_q_gain_ge_0.05_somewhere": bool(any(
                    b["arm_raw"]["mean_q_uar"] - b["arm_raw"]["mean_q_ord"] >= 0.05
                    for b in blocks)),
                "X2-P3_uar_longer": bool(all(
                    (b["arm_raw"]["mean_len_uar"] or 0) >= b["arm_raw"]["mean_len_ord"]
                    for b in blocks)),
                "X2-P4_uar_cov05_below_0.9_T4": bool(
                    ar["coverage"]["uar@0.05"] < 0.9),
                "X2-P4b_cal_cov05_T4": bt["arm_cal"]["coverage"]["ord@0.05"],
                "uar_identical_to_ord_everywhere": bool(all(
                    r["q_uar_raw"] == r["q_ord_raw"] and r["len_uar"] == r["len_ord"]
                    for b in blocks for r in b["per_instance"])),
            }
        report = {
            "protocol": {
                "question": ("X2: ordinary (min-size) vs uncertainty-aware (max-q) "
                             "sufficient reasons on FROZEN trees; completions "
                             "b'_j~Bernoulli(p_j); q(R)=P(literals true under the "
                             "completion model); same path-literal pool, exact "
                             "sufficiency both; raw arm pre-registered, cal arm "
                             "exploratory (isotonic on val carve-out, pos_weight "
                             "distortion check)"),
                "predictions_pre_registered": [
                    "X2-P1 >=50% of ORD reasons at K=4 T contain a conf<0.8 literal",
                    "X2-P2 UAR mean-q gain >= 0.05 somewhere; coverage UAR>ORD",
                    "X2-P3 mean |UAR| >= mean |ORD|",
                    "X2-P4 UAR coverage@0.05 < 0.90 at K=4 T (measured limit)",
                    "X2-P5 |fidelity_UAR - fidelity_ORD| < 0.10 on shared patterns",
                ],
                "B_montecarlo": args.B, "deltas": deltas,
                "frozen_models": ("readers (Adam, 40 epochs, val-carved train), "
                                  "CART T (depth12) on labels, patch-logreg + "
                                  "distilled CART D (depth4); no refit between "
                                  "explanation methods"),
                "note": ("literal confidences per arm; has_part and unobserved-part "
                         "attributes structural (conf 1); UAR exhaustive over path "
                         "subsets (exact max-q); calibrated arm uses val-split "
                         "isotonic P(gt bit | score), min 10 val samples"),
            },
            "blocks": blocks,
            "prereg_verdicts": verdicts,
        }
        Path(out_path).write_text(json.dumps(report, indent=2))
        print(f"saved: {out_path}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
