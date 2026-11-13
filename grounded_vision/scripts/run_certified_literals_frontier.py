"""
Script: run_certified_literals_frontier.py
==========================================
Round 3 (2026-11-08): the full operating frontier of the certified-
literal mechanism, with the statistical hygiene the review asked for.

Additions over run_certified_literals_cub.py:
- tau grid (default 0.5..0.95) -> coverage-vs-correctness frontier per K;
- Bonferroni-corrected certification (alpha = 0.05 / m tested literals)
  reported alongside raw Wilson bounds;
- per-literal prevalence and LIFT over the majority-polarity baseline
  on the calibration split;
- class-leakage audit on test: normalized mutual information between
  each literal's assertion and the species label (the yellow-bill
  species-detector should top this list);
- mechanism B v2 (exclusion semantics fix): the kept assignment now
  includes EVERY feature whose (concept, bit-from-phi) literal is
  certified -- including negatives of unread concepts (unknown->0
  encoding), which v1 could not hold, understating exclusion coverage.
- optional --explicit-classes for alternative species subsets.

PRE-REGISTERED PREDICTIONS (before the run; do not edit):
  PF1 K=4: coverage non-decreasing as tau falls, reaching 30-60% at
     tau=0.5; P(all|cov) >= 80% down to tau=0.6, possibly dipping at 0.5.
  PF2 K=10: at tau=0.5 coverage 5-20% with P|cov >= 50%; ~0 at tau>=0.8.
  PF3 K=20: coverage <= 5% at every tau (k~7 reasons need present
     literals, which only certify at low tau).
  PF4 Bonferroni removes no literal used by the K=4 tau=0.9 mechanism.
  PF5 every tau=0.8-certified negative literal has lift > 0 over its
     concept's absence baseline.
  PF6 bill_color=yellow|pos has the top class-MI among attribute literals
     (species-detector signature).
  PF7 mechanism B v2 coverage >= mechanism A coverage at every (K, tau).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend


def wilson_z(k, n, z):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def norm_ppf(q):
    # Abramowitz-Stegun inverse normal (avoids a scipy dependency)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        ql = math.sqrt(-2 * math.log(q))
        return (((((c[0]*ql+c[1])*ql+c[2])*ql+c[3])*ql+c[4])*ql+c[5]) / \
               ((((d[0]*ql+d[1])*ql+d[2])*ql+d[3])*ql+1)
    if q > ph:
        ql = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*ql+c[1])*ql+c[2])*ql+c[3])*ql+c[4])*ql+c[5]) / \
               ((((d[0]*ql+d[1])*ql+d[2])*ql+d[3])*ql+1)
    ql = q - 0.5
    r = ql * ql
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*ql / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


class FastBinaryTree:
    def __init__(self, dt):
        t = dt.tree_
        self.f = t.feature.astype(np.int64)
        self.l = t.children_left.astype(np.int64)
        self.r = t.children_right.astype(np.int64)
        self.pred = np.array(
            [int(dt.classes_[int(np.argmax(t.value[i][0]))])
             for i in range(t.node_count)], dtype=np.int64)

    def forces(self, assign, target):
        stack = [0]
        reached = False
        while stack:
            nid = stack.pop()
            if self.l[nid] == -1:
                if self.pred[nid] != target:
                    return False
                reached = True
            else:
                f = int(self.f[nid])
                if f in assign:
                    stack.append(int(self.r[nid] if assign[f] else self.l[nid]))
                else:
                    stack.append(int(self.l[nid]))
                    stack.append(int(self.r[nid]))
        return reached


def nmi(bit, label):
    """Normalized mutual information I(bit;class)/H(class)."""
    n = len(bit)
    from collections import Counter
    cb, cl = Counter(bit), Counter(label)
    cj = Counter(zip(bit, label))
    mi = sum(v / n * math.log2((v / n) / (cb[b] / n * cl[l] / n))
             for (b, l), v in cj.items())
    hc = -sum(v / n * math.log2(v / n) for v in cl.values())
    return mi / hc if hc > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="4,10,20")
    ap.add_argument("--explicit-classes", default=None,
                    help="comma list of class ids; overrides --classes subsets")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--taus", default="0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.explicit_classes:
        subsets = [sorted(int(c) for c in args.explicit_classes.split(","))]
    else:
        classes = [int(c) for c in args.classes.split(",")]
        subsets = [list(range(K)) for K in classes]
    taus = [float(t) for t in args.taus.split(",")]
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    dtr = np.load(Path(args.cache_dir) / "train_features.npz")
    dte = np.load(Path(args.cache_dir) / "test_features.npz")
    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=args.seed)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=args.seed)
    vocab = ds_train.bound_attributes
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=list(vocab))
    feat_names = compiler.feature_names
    fidx = compiler.feature_to_idx
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    part_bits = [n for n in feat_names if n.startswith("has_part(")]

    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])
    print(f"--> pooling train {len(tr_rows)} / test {len(te_rows)}", flush=True)

    def pooled(rows, fmaps):
        out = {}
        fm = torch.from_numpy(fmaps.astype(np.float32))
        for i, row in enumerate(rows):
            sid, _meta_sid, boxes, vis, _attrs = row
            regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                    for p in ds_train.part_names if vis.get(p, 0.0) > 0.5 and p in boxes]
            if regs:
                p = pool_region_features(fm[i:i + 1], regs)
                out[int(sid)] = {r.part_name: p[r.id].numpy() for r in regs}
        return out

    tr_pooled = pooled(tr_rows, dtr["feature_maps"])
    te_pooled = pooled(te_rows, dte["feature_maps"])

    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    te_gt = {int(m[0]): m[4] for m in te_rows}
    te_vis = {int(m[0]): m[3] for m in te_rows}
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

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
    print(f"--> readers trained on {len(fit_sids)} imgs (val {len(val_sids)})",
          flush=True)

    def prob_attr(j, vec):
        with torch.no_grad():
            t = torch.from_numpy(vec.reshape(1, -1)).float()
            return float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])

    def concept_probs(pooled):
        d = {}
        for part, vec in pooled.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                p = prob_attr(j, vec)
                d[f"{a.part}:{a.attr_name}={a.value}"] = (1 if p >= 0.5 else 0, p)
        for n in part_bits:
            d[n] = (1, 1.0)
        return d

    def vec_from(d):
        return np.array([d.get(n, (0, 0.0))[0] for n in feat_names], dtype=np.int64)

    def gt_bits(gt, vis):
        bits = {f"{a.part}:{a.attr_name}={a.value}": int(gt[j])
                for j, a in enumerate(vocab)}
        for n in part_bits:
            p = n[len("has_part("):-1]
            bits[n] = 1 if vis.get(p, 0.0) > 0.5 else 0
        return bits

    tr_cp = {sid: concept_probs(tr_pooled[sid]) for sid in sids}
    te_cp = {sid: concept_probs(te_pooled[sid]) for sid in te_pooled.keys()}

    # ---- literal certification on VAL + prevalence/lift ----------------------
    lit_stat, prev_stat = {}, {}  # (name,pol)->[c,n]; name->[pos_count, n_read]
    for sid in val_sids:
        gtb = gt_bits(tr_gt[sid], {p: 1.0 for p in tr_pooled[sid]})
        for name, (bit, _p) in tr_cp[sid].items():
            lit_stat.setdefault((name, bit), [0, 0])
            lit_stat[(name, bit)][1] += 1
            lit_stat[(name, bit)][0] += int(gtb.get(name) == bit)
            if not name.startswith("has_part"):
                prev_stat.setdefault(name, [0, 0])
                prev_stat[name][1] += 1
                prev_stat[name][0] += gtb.get(name, 0)
    tested = {k: v for k, v in lit_stat.items() if v[1] >= args.min_support}
    m_tests = len(tested)
    z_raw = norm_ppf(1 - 0.05 / 2)
    z_bonf = norm_ppf(1 - 0.05 / (2 * max(1, m_tests)))
    print(f"--> {m_tests} literals tested (support>={args.min_support}); "
          f"Bonferroni z={z_bonf:.2f} vs raw z={z_raw:.2f}", flush=True)

    cert_table = {}
    for (name, pol), (c0, n0) in sorted(tested.items()):
        w_raw = wilson_z(c0, n0, z_raw)
        w_bon = wilson_z(c0, n0, z_bonf)
        base = None
        if not name.startswith("has_part") and name in prev_stat:
            p_pos = prev_stat[name][0] / prev_stat[name][1]
            base = p_pos if pol else 1 - p_pos
        rate = c0 / n0
        cert_table[f"{name}|{'pos' if pol else 'neg'}"] = {
            "n_val": n0, "rate": rate,
            "wilson_lo": w_raw[0], "wilson_lo_bonferroni": w_bon[0],
            "baseline_majority": base,
            "lift": (rate - base) if base is not None else None,
            "certified_at": {str(t): bool(w_raw[0] >= t) for t in taus},
            "certified_at_bonferroni": {str(t): bool(w_bon[0] >= t) for t in taus},
        }
    n_pos_cert = sum(1 for k, v in cert_table.items()
                     if k.endswith("|pos") and not k.startswith("has_part")
                     and any(v["certified_at"].values()))
    print(f"--> attribute-POSITIVE literals certified (raw, any tau): {n_pos_cert}",
          flush=True)

    # ---- per-subset evaluation -----------------------------------------------
    results = []
    for subset in subsets:
        sub = set(subset)
        tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] in sub]
        te_ids = [s for s in te_pooled.keys() if sid_to_label_te[s] in sub]
        X_fit = np.stack([vec_from(tr_cp[s]) for s in tr_ids])
        y_fit = np.array([sid_to_label_tr[s] for s in tr_ids])
        dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                    random_state=42).fit(X_fit, y_fit)
        head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                                   feature_names=feat_names)
        head.clf, head.is_fitted = dt, True
        backend = PyXAIBackend(head)
        fast = FastBinaryTree(dt)

        # class-leakage NMI on this subset's test images
        leakage = {}
        bits_by_feat = defaultdict(list); labels = []
        for sid in te_ids:
            xp = vec_from(te_cp[sid])
            labels.append(sid_to_label_te[sid])
            for fi in range(len(feat_names)):
                bits_by_feat[fi].append(int(xp[fi]))
        for fi, name in enumerate(feat_names):
            if name.startswith("has_part"):
                continue
            for pol in (0, 1):
                b = [1 if x == pol else 0 for x in bits_by_feat[fi]]
                if sum(b) >= 5:
                    leakage[f"{name}|{'pos' if pol else 'neg'}"] = \
                        round(nmi(b, labels), 3)

        mechA = {t: defaultdict(int) for t in taus}
        mechB = {t: defaultdict(int) for t in taus}
        for sid in te_ids:
            cp = te_cp[sid]
            xp = vec_from(cp)
            gtb = gt_bits(te_gt[sid], te_vis[sid])
            pred = int(dt.predict(xp.reshape(1, -1))[0])
            exp = backend.explain_sufficient(xp, pred, sample_id=sid)
            lits = [(feat_names[c.feature_idx], 1 if c.operator != "<=" else 0)
                    for c in exp.conditions]
            allc = all(gtb.get(n) == p for n, p in lits)
            for t in taus:
                okA = all(cert_table.get(f"{n}|{'pos' if p else 'neg'}", {}
                                         ).get("certified_at", {}).get(str(t), False)
                          for n, p in lits)
                if okA:
                    mechA[t]["cov"] += 1; mechA[t]["k"] += len(lits)
                    mechA[t]["allc"] += allc
                    mechA[t]["bound"] += max(0.0, 1 - sum(
                        1 - cert_table[f"{n}|{'pos' if p else 'neg'}"]["wilson_lo"]
                        for n, p in lits))
                # B v2: every feature whose (concept, phi-bit) literal certifies
                kept = {fi: int(xp[fi]) for fi in range(len(feat_names))
                        if cert_table.get(
                            f"{feat_names[fi]}|{'pos' if xp[fi] else 'neg'}", {}
                        ).get("certified_at", {}).get(str(t), False)}
                if kept and fast.forces(kept, pred):
                    for f in list(kept.keys()):
                        smaller = {k: v for k, v in kept.items() if k != f}
                        if smaller and fast.forces(smaller, pred):
                            kept = smaller
                    mechB[t]["cov"] += 1
                    mechB[t]["kept"] += len(kept)
                    mechB[t]["allc"] += all(gtb.get(feat_names[f]) == b
                                            for f, b in kept.items())
                    mechB[t]["bound"] += max(0.0, 1 - sum(
                        1 - cert_table[
                            f"{feat_names[f]}|{'pos' if b else 'neg'}"]["wilson_lo"]
                        for f, b in kept.items()))

        entry = {
            "classes": subset, "n_train": len(tr_ids), "n_test": len(te_ids),
            "tree_test_acc": float(np.mean([
                int(dt.predict(vec_from(te_cp[s]).reshape(1, -1))[0])
                == sid_to_label_te[s] for s in te_ids])),
            "default": {"mean_k": None, "p_all": None},
            "frontier": {},
            "leakage_nmi_top": dict(sorted(leakage.items(),
                                           key=lambda kv: -kv[1])[:10]),
        }
        ks, alls = [], []
        for sid in te_ids:
            xp = vec_from(te_cp[sid])
            gtb = gt_bits(te_gt[sid], te_vis[sid])
            pred = int(dt.predict(xp.reshape(1, -1))[0])
            exp = backend.explain_sufficient(xp, pred, sample_id=sid)
            lits = [(feat_names[c.feature_idx], 1 if c.operator != "<=" else 0)
                    for c in exp.conditions]
            ks.append(len(lits))
            alls.append(all(gtb.get(n) == p for n, p in lits))
        entry["default"] = {"mean_k": float(np.mean(ks)),
                            "p_all": float(np.mean(alls))}
        for t in taus:
            a, b = mechA[t], mechB[t]
            entry["frontier"][str(t)] = {
                "A_minimal_reason": {
                    "coverage": a["cov"] / len(te_ids), "n_covered": a["cov"],
                    "p_all_given_cov": (a["allc"] / a["cov"]) if a["cov"] else None,
                    "mean_k": (a["k"] / a["cov"]) if a["cov"] else None,
                    "mean_union_bound": (a["bound"] / a["cov"]) if a["cov"] else None},
                "B_exclusion_v2": {
                    "coverage": b["cov"] / len(te_ids), "n_covered": b["cov"],
                    "p_all_given_cov": (b["allc"] / b["cov"]) if b["cov"] else None,
                    "mean_kept": (b["kept"] / b["cov"]) if b["cov"] else None,
                    "mean_union_bound": (b["bound"] / b["cov"]) if b["cov"] else None},
            }
        results.append(entry)
        print(f"--> classes {subset}: default k {entry['default']['mean_k']:.2f} "
              f"P(all) {entry['default']['p_all']:.3f} | frontier " +
              " ; ".join(
                  f"tau={t}: A {entry['frontier'][str(t)]['A_minimal_reason']['coverage']:.3f}"
                  f"/{entry['frontier'][str(t)]['A_minimal_reason']['p_all_given_cov']}"
                  f" B {entry['frontier'][str(t)]['B_exclusion_v2']['coverage']:.3f}"
                  f"/{entry['frontier'][str(t)]['B_exclusion_v2']['p_all_given_cov']}"
                  for t in taus), flush=True)

    report = {
        "protocol": {
            "question": ("Certified-literal operating frontier: coverage vs "
                         "P(all clauses correct | covered) over tau and K, with "
                         "Bonferroni correction, majority-baseline lift, "
                         "class-leakage NMI, and exclusion-semantics fix (B v2)"),
            "predictions_pre_registered": [
                "PF1 K=4: coverage 30-60% at tau=0.5, P|cov >= 80% to tau=0.6",
                "PF2 K=10: tau=0.5 coverage 5-20% P|cov >= 50%; ~0 at tau>=0.8",
                "PF3 K=20: coverage <= 5% at every tau",
                "PF4 Bonferroni removes no literal used by K=4 tau=0.9 mechanism",
                "PF5 every tau=0.8-certified negative literal has lift > 0",
                "PF6 bill_color=yellow|pos tops class-leakage NMI",
                "PF7 B v2 coverage >= A coverage at every (K, tau)",
            ],
            "splits": ("calibration = 15% validation sub-split of train; "
                       "test = official held-out split (disjoint); trees fit on "
                       "the 85% train sub-split only"),
            "multiplicity": f"Bonferroni alpha=0.05/{m_tests} (z={z_bonf:.3f})",
            "tree": "CART depth 12, leaf 4, seed 42, full 36 concepts",
            "seed": args.seed,
            "explicit_classes": args.explicit_classes,
            "cache": "eval-transform (flip-free)",
        },
        "literal_certification": cert_table,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
