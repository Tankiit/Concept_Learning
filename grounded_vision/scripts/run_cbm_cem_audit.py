"""
Script: run_cbm_cem_audit.py
============================
CBM/CEM/Boolean audit (2026-11-08): what does a concept explanation
actually tell us?  Frozen pipeline identical to the committed X2 /
guarantee-scaling protocol (oracle GT part boxes, DINOv3 cached feature
maps, concept readers on pooled region features), with three task heads
over the SAME concept vocabulary:

  standard CBM   multinomial logistic head on continuous concept scores
  standard CEM   task head on learned per-concept EMBEDDINGS
                 (shared per-part trunk 384->128->ReLU->16, per-concept
                 aux concept head; joint CE + BCE loss).  NOT certified:
                 embeddings can encode information beyond the concept
                 labels (that is CEM's design), so exact Boolean
                 certificates are reserved for the deployed Boolean
                 model.
  Boolean CBM    deployed = CART on thresholded reader bits
                 (Setting T: fitted on labels; Setting D: distilled
                 from the frozen mean-patch linear head).

AUDIT (deployed Boolean model; all test images at K<=10, seeded
subsample at larger K).  Per image, measured SEPARATELY:

  task correctness   tree prediction == label
  logical sufficiency native backward-elimination minimal sufficient
                     reason, certificate from TreeSufficiencyVerifier,
                     re-verified by an independent exact reachable-leaf
                     DFS, plus real-PyXAI cross-check on an exported
                     subset (scripts/pyxai_audit_local.py)
  premise correctness every signed literal vs the CUB image-level
                     annotation: supported / refuted; literals over
                     UNOBSERVED parts additionally flagged unresolved
                     (reader never read the region, so the premise is
                     not grounded in the image)
  length             number of literals

MAIN OUTPUT: P(correct AND verified AND all-premises-supported), with
every component reported separately so a low joint rate decomposes.

SELECTION (hypothesis leg): same predictor, same images, same
path-literal pool; exact exhaustive enumeration over <=2^12 subsets:
  default          min-size sufficient subset
  confidence       max calibrated q(R) sufficient subset
  annotation-oracle max supported-literals sufficient subset
Decision rule: continue selection research only if useful supported
alternatives exist (X2's uniqueness finding predicts no room).

CONTROLS / CONTEXT: majority-class constant classifier per K; present/
absent clause precision (attribute imbalance); per-concept reader test
accuracy; oracle concept interventions (tree at GT bits; CBM head at
GT-bit scores); CEM accuracy without embedding-level certification.

Pre-registered readouts (A1-A5), recorded before the run:
  A1 verified_frac == 1.0 everywhere (sanity; a failure is a bug).
  A2 P(all supported) < accuracy at every K>=4.
  A3 joint < min(P(correct), P(all supported)) at every setting.
  A4 selectors coincide on >= 95% of instances; oracle-gain <= 5%.
  A5 unresolved literals occur in reasons (category non-empty).
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
import torch.nn as nn
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
from grounded_vision.explanations.schema import VerificationStatus


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


class FastBinaryTree:
    """Exact reachable-leaf analysis for a binary-input sklearn tree
    (independent of TreeSufficiencyVerifier's interval arithmetic)."""

    def __init__(self, dt):
        t = dt.tree_
        self.f = t.feature.astype(np.int64)
        self.l = t.children_left.astype(np.int64)
        self.r = t.children_right.astype(np.int64)
        self.pred = np.array(
            [int(dt.classes_[int(np.argmax(t.value[i][0]))])
             for i in range(t.node_count)], dtype=np.int64)

    def forces(self, assign: dict, target: int) -> bool:
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
    if is_left:  # feature <= thr
        if fidx in leaf["lo"]:
            return thr <= leaf["lo"][fidx]
        return False
    else:       # feature > thr
        if fidx in leaf["hi"]:
            return thr >= leaf["hi"][fidx]
        return False


class MLPReader(nn.Module):
    """Stronger concept reader for the repeat run: shared per-part
    nonlinear trunk + per-concept linear head (same supervision)."""

    def __init__(self, parts, n_attrs, hidden=128):
        super().__init__()
        self.trunks = nn.ModuleDict(
            {p: nn.Sequential(nn.Linear(384, hidden), nn.ReLU()) for p in parts})
        self.heads = nn.ModuleDict(
            {f"attr_{j}": nn.Linear(hidden, 1) for j in range(n_attrs)})

    def logits(self, part, X, cols):
        Z = self.trunks[part](X)
        W = torch.cat([self.heads[f"attr_{j}"].weight for j in cols], dim=0)
        b = torch.cat([self.heads[f"attr_{j}"].bias for j in cols], dim=0)
        return Z @ W.T + b


class CEM(nn.Module):
    """Compact Concept Embedding Model: per-part trunks -> per-concept
    16-d embedding (concepts of a part share that part's trunk output,
    expanded to concept slots in vocab order) -> linear task head on the
    concatenated embeddings; per-concept aux heads supervised by
    annotation bits where the part is observed.  Unobserved parts enter
    the trunk as zero vectors."""

    def __init__(self, parts, concept_parts, n_classes, emb=16, hidden=128):
        super().__init__()
        self.trunks = nn.ModuleDict(
            {p: nn.Sequential(nn.Linear(384, hidden), nn.ReLU(),
                              nn.Linear(hidden, emb)) for p in parts})
        n_concepts = len(concept_parts)
        self.task_head = nn.Linear(n_concepts * emb, n_classes)
        self.aux = nn.ModuleDict(
            {f"c{j}": nn.Linear(emb, 1) for j in range(n_concepts)})
        self.n_concepts, self.emb, self.parts = n_concepts, emb, parts
        self.concept_parts = list(concept_parts)

    def embed(self, X_parts):
        E_p = {p: self.trunks[p](X_parts[p]) for p in self.parts}
        return torch.cat([E_p[part] for part in self.concept_parts], dim=1)

    def forward(self, X_parts):
        return self.task_head(self.embed(X_parts))

    def concept_logits(self, X_parts):
        E = self.embed(X_parts).view(-1, self.n_concepts, self.emb)
        return torch.cat([self.aux[f"c{j}"](E[:, j, :])
                          for j in range(self.n_concepts)], dim=1)


def enumerate_selectors(path_lits, leaves, target, lit_meta):
    """Exact exhaustive selection over sufficient subsets of the path
    literals.  lit_meta aligned with path_lits: per-literal (conf,
    supported, refuted, unresolved).  Returns None if no sufficient
    subset exists from the path (cannot happen when the native reason
    exists)."""
    neg_masks = []
    for lf in leaves:
        if lf["pred"] == target:
            continue
        mask = 0
        for i, (fidx, is_left, thr) in enumerate(path_lits):
            if literal_contradicts_leaf(fidx, is_left, thr, lf):
                mask |= (1 << i)
        if mask == 0:
            return None
        neg_masks.append(mask)
    m = len(path_lits)
    if m == 0:
        return None
    conf = np.array([mm[0] for mm in lit_meta])
    sup = np.array([1 if mm[1] else 0 for mm in lit_meta])
    ref = np.array([1 if mm[2] else 0 for mm in lit_meta])
    unr = np.array([1 if mm[3] else 0 for mm in lit_meta])
    logc = -np.log(np.clip(conf, 1e-12, 1.0))

    S = np.arange(1, 1 << m, dtype=np.int64)
    bits = ((S[:, None] >> np.arange(m)) & 1).astype(bool)      # (2^m-1, m)
    mbits = ((np.array(neg_masks, dtype=np.int64)[:, None] >> np.arange(m)) & 1
             ).astype(bool)                                       # (M, m)
    block = (mbits[:, None, :] & bits[None, :, :]).any(2)         # (M, 2^m-1)
    suff = block.all(0)
    if not suff.any():
        return None
    idx = np.where(suff)[0]
    size = bits[idx].sum(1)
    logq = bits[idx] @ logc
    n_sup = bits[idx] @ sup
    n_ref = bits[idx] @ ref
    n_unr = bits[idx] @ unr
    minsz = size.min()
    default_rows = idx[size == minsz]
    order_conf = np.lexsort((size, -logq))       # primary max q, then shorter
    conf_row = idx[order_conf[0]]
    order_orc = np.lexsort((size, n_ref, -n_sup))  # primary max supported
    orc_row = idx[order_orc[0]]
    dflt_row = default_rows[0]

    def pack(row):
        sel = bits[row]
        return {
            "size": int(sel.sum()),
            "q": float(np.exp(-float(bits[row] @ logc))),
            "n_supported": int(bits[row] @ sup),
            "n_refuted": int(bits[row] @ ref),
            "n_unresolved": int(bits[row] @ unr),
            "all_supported": bool(int(bits[row] @ ref) == 0),
            "all_supported_grounded": bool(int(bits[row] @ ref) == 0
                                           and int(bits[row] @ unr) == 0),
        }

    return {
        "default": pack(dflt_row),
        "confidence": pack(conf_row),
        "oracle": pack(orc_row),
        "n_min_size_subsets": int((size == minsz).sum()),
        "default_is_unique_min": bool((size == minsz).sum() == 1),
        "default_eq_conf": bool(dflt_row == conf_row),
        "default_eq_oracle": bool(dflt_row == orc_row),
        "n_sufficient_subsets": int(suff.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="4,10,200")
    ap.add_argument("--class-ids", default=None,
                    help="comma-separated explicit class labels; overrides the "
                         "label<K first-K selection (subset-sensitivity draws)")
    ap.add_argument("--settings", default="T,D")       # audited tree settings
    ap.add_argument("--reader", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cem-epochs", type=int, default=60)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--n-reason-samples", type=int, default=600)
    ap.add_argument("--select-samples", type=int, default=200)
    ap.add_argument("--pyxai-export", default=None,
                    help="write a PyXAI cross-check payload (seed 42 only)")
    ap.add_argument("--pyxai-per-setting", type=int, default=118)
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    settings = args.settings.split(",")
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
    fidx = compiler.feature_to_idx
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    part_feat_idx = [fidx[n] for n in feat_names if n.startswith("has_part(")]
    feat_part = {}
    for a in vocab:
        feat_part[fidx[f"{a.part}:{a.attr_name}={a.value}"]] = a.part
    for n in feat_names:
        if n.startswith("has_part("):
            feat_part[fidx[n]] = n[len("has_part("):-1]

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
    te_gt = {int(m[0]): m[4] for m in te_rows}
    te_vis = {int(m[0]): m[3] for m in te_rows}
    tr_vis = {int(m[0]): m[3] for m in tr_rows}
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}
    te_order = {int(m[0]): i for i, m in enumerate(te_rows)}
    tr_order = {int(m[0]): i for i, m in enumerate(tr_rows)}
    tr_fm, te_fm = dtr["feature_maps"], dte["feature_maps"]

    def gt_vec(gt, vis):
        bits = {f"{a.part}:{a.attr_name}={a.value}": float(gt[j])
                for j, a in enumerate(vocab)}
        for n in feat_names:
            if n.startswith("has_part("):
                p = n[len("has_part("):-1]
                bits[n] = 1.0 if vis.get(p, 0.0) > 0.5 else 0.0
        return np.array([bits.get(n, 0.0) for n in feat_names], dtype=np.int64)

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

        # ---- concept readers (frozen) --------------------------------------
        if args.reader == "linear":
            reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
            params = reader.parameters()
        else:
            reader = MLPReader(sorted(attr_by_part.keys()), len(vocab))
            params = reader.parameters()
        opt = torch.optim.Adam(params, lr=args.lr)
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
            for part, (X, y, pw, cols) in region_data.items():
                if args.reader == "linear":
                    W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
                    b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
                    logits = X @ W.T + b
                else:
                    logits = reader.logits(part, X, cols)
                loss = loss + F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"--> seed {seed}: {args.reader} readers trained", flush=True)

        def scores_map(parts):
            d = {}
            for part, vec in parts.items():
                for j in attr_by_part.get(part, []):
                    a = vocab[j]
                    t = torch.from_numpy(vec.reshape(1, -1)).float()
                    with torch.no_grad():
                        if args.reader == "linear":
                            lg = reader.classifiers[f"attr_{j}"](t)
                        else:
                            lg = reader.logits(part, t, [j])
                    d[f"{a.part}:{a.attr_name}={a.value}"] = float(torch.sigmoid(lg)[0, 0])
            return d

        # ---- calibrated literal probabilities (val carve-out) --------------
        val_scores = defaultdict(list)
        for sid in val_sids:
            d = scores_map(tr_pooled[sid])
            for name, sc in d.items():
                fi = fidx[name]
                j = next(k for k, a in enumerate(vocab)
                         if f"{a.part}:{a.attr_name}={a.value}" == name)
                val_scores[fi].append((sc, float(tr_gt[sid][j])))
        isotonic = {}
        for fi, pairs in val_scores.items():
            if len(pairs) < 10:
                continue
            xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
            if ys.min() == ys.max():
                isotonic[fi] = ("const", float(ys[0]))
            else:
                isotonic[fi] = ("ir", IsotonicRegression(out_of_bounds="clip").fit(xs, ys))

        def cal_prob(fi, s):
            if fi not in isotonic:
                return s
            kind, obj = isotonic[fi]
            return float(obj) if kind == "const" else float(obj.predict([s])[0])

        def vecs(parts, vis):
            """(bits_T, bits_D, score vector incl. has_part, calibrated)"""
            d = scores_map(parts)
            sc = np.zeros(len(feat_names), dtype=np.float64)
            sc_cal = np.zeros(len(feat_names), dtype=np.float64)
            for name, p in d.items():
                fi = fidx[name]
                sc[fi] = p
                sc_cal[fi] = cal_prob(fi, p)
            for fi in part_feat_idx:
                bit = 1.0 if vis.get(feat_part[fi], 0.0) > 0.5 else 0.0
                sc[fi] = bit
                sc_cal[fi] = bit
            bthr = (sc >= 0.5).astype(np.int64)
            bT = bthr.copy()
            bT[part_feat_idx] = 1
            bD = bthr.copy()
            bD[part_feat_idx] = 0
            return bT, bD, sc, sc_cal

        tr_bT, tr_bD, tr_sc, tr_scc = {}, {}, {}, {}
        for s in sorted(fit_sids):
            a, b, c, e = vecs(tr_pooled[s], tr_vis[s])
            tr_bT[s], tr_bD[s], tr_sc[s], tr_scc[s] = a, b, c, e
        te_bT, te_bD, te_sc, te_scc = {}, {}, {}, {}
        for s in te_pooled.keys():
            a, b, c, e = vecs(te_pooled[s], te_vis[s])
            te_bT[s], te_bD[s], te_sc[s], te_scc[s] = a, b, c, e

        # ---- concept quality on test (observed parts only) -----------------
        concept_quality = {}
        for j, a in enumerate(vocab):
            name = f"{a.part}:{a.attr_name}={a.value}"
            fi = fidx[name]
            ok = tot = present = 0
            for s in te_pooled.keys():
                if a.part not in te_pooled[s]:
                    continue
                bit = int(te_sc[s][fi] >= 0.5)
                ok += int(bit == int(te_gt[s][j]))
                present += int(int(te_gt[s][j]) == 1)
                tot += 1
            concept_quality[name] = {
                "n_observed": tot,
                "accuracy": round(ok / tot, 4) if tot else None,
                "gt_present_rate": round(present / tot, 4) if tot else None,
            }
        accs = [v["accuracy"] for v in concept_quality.values() if v["accuracy"] is not None]
        concept_quality_summary = {
            "n_concepts": len(accs),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "median_accuracy": round(float(np.median(accs)), 4),
            "min_accuracy": round(float(np.min(accs)), 4),
        }
        print(f"--> seed {seed}: reader concept accuracy mean "
              f"{concept_quality_summary['mean_accuracy']:.3f} min "
              f"{concept_quality_summary['min_accuracy']:.3f}", flush=True)

        out_path = Path(args.out if seed == 42
                        else str(Path(args.out)).replace(".json", f"_seed{seed}.json"))
        blocks = []
        pyxai_payload = []       # (block_name, instance dict)
        tree_store = {}          # block_name -> sklearn tree
        for K in classes:
            sel = class_ids if class_ids is not None else range(int(K))
            tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] in sel]
            te_ids = [s for s in te_pooled.keys() if sid_to_label_te[s] in sel]
            y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
            y_te = np.array([sid_to_label_te[s] for s in te_ids])

            # ================= context heads =================================
            majority = int(np.bincount(y_tr).argmax())
            const_acc = float(np.mean(y_te == majority))

            # standard CBM: logistic head on continuous concept scores
            Xcbm_tr = np.stack([tr_sc[s] for s in tr_ids])
            Xcbm_te = np.stack([te_sc[s] for s in te_ids])
            cbm = LogisticRegression(max_iter=3000, C=1.0,
                                     random_state=42).fit(Xcbm_tr, y_tr)
            cbm_acc = float(np.mean(cbm.predict(Xcbm_te) == y_te))
            # oracle concept intervention on the CBM: scores -> GT bits
            Xgt_te = np.stack([gt_vec(te_gt[s], te_vis[s]).astype(np.float64)
                               for s in te_ids])
            cbm_gt_acc = float(np.mean(cbm.predict(Xgt_te) == y_te))

            # standard CEM (torch; per-part trunks + concept embeddings)
            torch.manual_seed(seed * 10 + K)
            parts = sorted(attr_by_part.keys())
            concept_parts = [a.part for a in vocab]
            cem = CEM(parts, concept_parts, n_classes=int(K))
            optc = torch.optim.Adam(cem.parameters(), lr=1e-3, weight_decay=1e-4)
            cem_parts_tr = {p: torch.from_numpy(np.stack([
                tr_pooled[s].get(p, np.zeros(384, dtype=np.float32))
                for s in tr_ids])).float() for p in parts}
            cem_parts_te = {p: torch.from_numpy(np.stack([
                te_pooled[s].get(p, np.zeros(384, dtype=np.float32))
                for s in te_ids])).float() for p in parts}
            y_tr_t = torch.from_numpy(y_tr).long()
            gt_bits_tr = torch.from_numpy(np.stack(
                [tr_gt[s] for s in tr_ids])).float()
            obs_mask = torch.from_numpy(np.stack([
                np.array([1.0 if a.part in tr_pooled[s] else 0.0
                          for a in vocab]) for s in tr_ids])).float()
            n_tr = len(tr_ids)
            bs = 512
            for ep in range(args.cem_epochs):
                cem.train()
                pr = np.random.RandomState(seed * 100 + ep).permutation(n_tr)
                for i0 in range(0, n_tr, bs):
                    idx = pr[i0:i0 + bs]
                    xb = {p: cem_parts_tr[p][idx] for p in parts}
                    logits = cem(xb)
                    cl = cem.concept_logits(xb)
                    m = obs_mask[idx].unsqueeze(1)
                    loss = F.cross_entropy(logits, y_tr_t[idx]) \
                        + (F.binary_cross_entropy_with_logits(cl, gt_bits_tr[idx],
                                                              reduction="none") * m
                           ).sum() / m.sum().clamp(min=1.0)
                    optc.zero_grad(); loss.backward(); optc.step()
            cem.eval()
            with torch.no_grad():
                cem_pred = torch.cat([
                    cem({p: cem_parts_te[p][i:i + bs] for p in parts}).argmax(dim=1)
                    for i in range(0, len(te_ids), bs)]).numpy()
            cem_acc = float(np.mean(cem_pred == y_te))

            # ================= deployed Boolean models =======================
            X_tr_T = np.stack([tr_bT[s] for s in tr_ids])
            X_te_T = np.stack([te_bT[s] for s in te_ids])
            dtT = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                         random_state=42).fit(X_tr_T, y_tr)
            teacherT = LogisticRegression(max_iter=2000, C=1.0,
                                          random_state=42).fit(X_tr_T, y_tr)
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

            audit_settings = []
            if "T" in settings:
                audit_settings.append(("T_tree_head", dtT, X_te_T, te_bT,
                                       teacherT.predict(X_te_T)))
            if "D" in settings and K <= 20:
                audit_settings.append(("D_distilled", dtD, X_te_D, te_bD,
                                       patch_head.predict(Xp_te)))

            for sname, dt, X_te, bits_setting, teacher_pred in audit_settings:
                block_name = f"{sname}_K{K}"
                head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                                           feature_names=feat_names)
                head.clf, head.is_fitted = dt, True
                back = PyXAIBackend(head)
                fast = FastBinaryTree(dt)
                leaves = leaf_bounds(dt)
                tree_pred = dt.predict(X_te)
                n_te = len(te_ids)

                if n_te <= args.n_reason_samples:
                    sel = list(te_ids)
                else:
                    idx = np.sort(np.random.RandomState(7).permutation(n_te)
                                  [: args.n_reason_samples])
                    sel = [te_ids[int(i)] for i in idx]
                sel_set = set(sel)
                if len(sel) > args.select_samples:
                    sidx = np.sort(np.random.RandomState(11).permutation(len(sel))
                                   [: args.select_samples])
                    sel_sel = [sel[int(i)] for i in sidx]
                else:
                    sel_sel = list(sel)
                sel_sel_set = set(sel_sel)

                rows = []
                pyxai_local = []
                for i, s in enumerate(te_ids):
                    if s not in sel_set:
                        continue
                    v = bits_setting[s]
                    c = int(tree_pred[i])
                    xg = gt_vec(te_gt[s], te_vis[s])
                    vis = te_vis[s]
                    exp = back.explain_sufficient(v, c, sample_id=s)
                    verified_cert = (exp.certificate.status
                                     == VerificationStatus.VERIFIED)
                    lits = [(int(cc.feature_idx), 1 if cc.operator != "<=" else 0)
                            for cc in exp.conditions]
                    # independent re-verification: exact reachable-leaf DFS
                    verified_dfs = fast.forces({f: b for f, b in lits}, c)
                    lit_rows, n_ref, n_unr, n_sup = [], 0, 0, 0
                    for f, b in lits:
                        part = feat_part[f]
                        observed = vis.get(part, 0.0) > 0.5
                        supported = bool(int(xg[f]) == b)
                        unresolved = (not observed) and f not in part_feat_idx
                        n_sup += supported
                        n_ref += (not supported)
                        n_unr += unresolved
                        lit_rows.append({
                            "feature": feat_names[f], "bit": b, "part": part,
                            "observed": observed, "gt_bit": int(xg[f]),
                            "supported": supported, "unresolved": unresolved,
                        })
                    all_sup = bool(len(lits) > 0 and n_ref == 0)
                    all_sup_grounded = bool(all_sup and n_unr == 0)
                    rec = {
                        "sid": int(s), "pred": c, "true": int(y_te[i]),
                        "correct": bool(c == int(y_te[i])),
                        "len": len(lits),
                        "verified_cert": bool(verified_cert),
                        "verified_dfs": bool(verified_dfs),
                        "n_supported": n_sup, "n_refuted": n_ref,
                        "n_unresolved": n_unr,
                        "all_supported": all_sup,
                        "all_supported_grounded": all_sup_grounded,
                        "literals": lit_rows,
                    }

                    if s in sel_sel_set:
                        steps = head.get_decision_path(v)
                        path_lits = [(st.feature_idx, st.is_left, st.threshold)
                                     for st in steps]
                        if path_lits:
                            lit_meta = []
                            for f, is_left, _thr in path_lits:
                                b = 0 if is_left else 1
                                p_conf = (te_scc[s][f] if b == 1
                                          else 1.0 - te_scc[s][f])
                                supported = bool(int(xg[f]) == b)
                                observed = vis.get(feat_part[f], 0.0) > 0.5
                                unresolved = (not observed) and f not in part_feat_idx
                                lit_meta.append((p_conf, supported,
                                                 not supported, unresolved))
                            rec["selection"] = enumerate_selectors(
                                path_lits, leaves, c, lit_meta)
                            if rec["selection"]:
                                rec["selection"]["default_matches_native"] = bool(
                                    rec["selection"]["default"]["size"] == len(lits))
                        if len(pyxai_local) < args.pyxai_per_setting \
                                and args.pyxai_export and seed == seeds[0]:
                            pyxai_local.append({
                                "sample_id": int(s), "x": [int(x) for x in v],
                                "prediction": c,
                                "native_reason_size": len(lits),
                                "native_reason_literals": [
                                    (f + 1) if b == 1 else -(f + 1)
                                    for f, b in lits],
                                "block": block_name,
                            })
                    rows.append(rec)

                n_r = len(rows)
                n_correct = sum(r["correct"] for r in rows)
                n_ver = sum(r["verified_cert"] and r["verified_dfs"] for r in rows)
                n_sup_all = sum(r["all_supported"] for r in rows)
                n_joint = sum(r["correct"] and r["verified_cert"]
                              and r["verified_dfs"] and r["all_supported"]
                              for r in rows)
                n_joint_g = sum(r["correct"] and r["verified_cert"]
                                and r["verified_dfs"]
                                and r["all_supported_grounded"] for r in rows)
                sel_rows = [r for r in rows if r.get("selection")]
                uniq = [r["selection"]["default_is_unique_min"]
                        for r in sel_rows if r["selection"]]
                oracle_gain = sum(
                    1 for r in sel_rows
                    if r["selection"]["oracle"]["all_supported"]
                    and not r["all_supported"])
                conf_gain = sum(
                    1 for r in sel_rows
                    if r["selection"]["confidence"]["all_supported"]
                    and not r["all_supported"])
                pres_t = sum(1 for r in rows for l in r["literals"] if l["bit"] == 1)
                pres_c = sum(1 for r in rows for l in r["literals"]
                             if l["bit"] == 1 and l["supported"])
                abs_t = sum(1 for r in rows for l in r["literals"] if l["bit"] == 0)
                abs_c = sum(1 for r in rows for l in r["literals"]
                            if l["bit"] == 0 and l["supported"])
                # Boolean intervention: deployed tree at GT-bit inputs
                tree_gt_acc = float(np.mean([
                    dt.predict(gt_vec(te_gt[s], te_vis[s]).reshape(1, -1))[0]
                    == sid_to_label_te[s] for s in te_ids]))

                block = {
                    "setting": sname, "K": K, "seed": seed,
                    "reader": args.reader,
                    "n_test": n_te, "n_audited": n_r,
                    "context": {
                        "majority_acc": round(const_acc, 4),
                        "cbm_acc": round(cbm_acc, 4),
                        "cbm_gt_intervention_acc": round(cbm_gt_acc, 4),
                        "cem_acc": round(cem_acc, 4),
                        "tree_acc_full": round(
                            float(np.mean(tree_pred == y_te)), 4),
                        "tree_acc_audited": round(n_correct / n_r, 4),
                        "tree_gt_bits_acc": round(tree_gt_acc, 4),
                        "teacher_agreement": round(
                            float(np.mean(teacher_pred == tree_pred)), 4),
                    },
                    "audit": {
                        "accuracy": round(n_correct / n_r, 4),
                        "verified_frac": round(n_ver / n_r, 4),
                        "mean_len": round(
                            float(np.mean([r["len"] for r in rows])), 3),
                        "p_all_supported": round(n_sup_all / n_r, 4),
                        "p_all_supported_wilson95": wilson(n_sup_all, n_r),
                        "p_joint_correct_verified_supported": round(
                            n_joint / n_r, 4),
                        "p_joint_wilson95": wilson(n_joint, n_r),
                        "p_joint_grounded": round(n_joint_g / n_r, 4),
                        "p_all_supported_given_correct": round(
                            sum(r["all_supported"] for r in rows if r["correct"])
                            / max(1, n_correct), 4),
                        "present_clause_precision": round(pres_c / pres_t, 4)
                        if pres_t else None,
                        "absent_clause_precision": round(abs_c / abs_t, 4)
                        if abs_t else None,
                        "frac_reasons_with_unresolved_literal": round(
                            sum(1 for r in rows if r["n_unresolved"] > 0) / n_r, 4),
                        "frac_literals_unresolved": round(
                            sum(r["n_unresolved"] for r in rows) /
                            max(1, sum(r["len"] for r in rows)), 4),
                    },
                    "selection": {
                        "n": len(sel_rows),
                        "frac_default_unique_min": round(
                            float(np.mean(uniq)), 4) if uniq else None,
                        "frac_default_eq_conf": round(float(np.mean(
                            [r["selection"]["default_eq_conf"] for r in sel_rows])), 4)
                        if sel_rows else None,
                        "frac_default_eq_oracle": round(float(np.mean(
                            [r["selection"]["default_eq_oracle"] for r in sel_rows])), 4)
                        if sel_rows else None,
                        "frac_default_matches_native": round(float(np.mean(
                            [r["selection"]["default_matches_native"]
                             for r in sel_rows if r["selection"]])), 4)
                        if any(r["selection"] for r in sel_rows) else None,
                        "oracle_gain_over_default": int(oracle_gain),
                        "confidence_gain_over_default": int(conf_gain),
                        "mean_n_sufficient_subsets": round(float(np.mean(
                            [r["selection"]["n_sufficient_subsets"]
                             for r in sel_rows if r["selection"]])), 2)
                        if any(r["selection"] for r in sel_rows) else None,
                    },
                    "per_instance": rows,
                }
                blocks.append(block)
                if pyxai_local:
                    pyxai_payload.extend(pyxai_local)
                    tree_store[block_name] = dt
                a = block["audit"]
                print(f"--> seed {seed} {sname} K={K}: acc {a['accuracy']:.3f} | "
                      f"len {a['mean_len']:.2f} | P(sup) {a['p_all_supported']:.3f} | "
                      f"P(joint) {a['p_joint_correct_verified_supported']:.3f} | "
                      f"P(joint,gnd) {a['p_joint_grounded']:.3f} | "
                      f"CBM {cbm_acc:.3f} CEM {cem_acc:.3f} maj {const_acc:.3f} | "
                      f"uniq {block['selection']['frac_default_unique_min']}",
                      flush=True)

        report = {
            "protocol": {
                "question": ("When the deployed Boolean CBM explains a prediction "
                             "with a small set of concepts: are the concepts "
                             "sufficient for its decision (logical sufficiency, "
                             "independently verified) and correct for the image "
                             "(premise support vs CUB image-level annotations)?"),
                "main_output": ("P(correct prediction AND verified sufficient "
                                "reason AND all premises annotation-supported), "
                                "with components reported separately"),
                "frozen_pipeline": ("DINOv3 cached features, oracle GT part "
                                    "boxes, concept readers on pooled region "
                                    "features (val-carved), CART T depth12 on "
                                    "labels / mean-patch linear teacher + "
                                    "distilled CART D depth4"),
                "cbm_head": "multinomial logistic on continuous concept scores",
                "cem_head": ("per-part trunk 384-128-16, concept embeddings -> "
                             "linear task head, aux concept heads, CE+BCE joint; "
                             "NOT certified (embeddings may encode beyond concept "
                             "labels; exact certificates reserved for the "
                             "deployed Boolean model)"),
                "premise_trichotomy": ("literal supported/refuted vs annotation; "
                                       "literals over unobserved parts flagged "
                                       "unresolved (reader never read the region)"),
                "selectors": ("default min-size / confidence max calibrated q / "
                              "annotation-oracle max supported; exact exhaustive "
                              "over <=2^12 path-literal subsets"),
                "prereg_readouts": [
                    "A1 verified_frac == 1.0 (sanity)",
                    "A2 P(all supported) < accuracy at every K>=4",
                    "A3 joint < min(P(correct), P(all supported))",
                    "A4 selectors coincide >= 95%; oracle-gain <= 5%",
                    "A5 unresolved literals occur in reasons",
                ],
                "reader": args.reader,
            },
            "concept_quality": concept_quality,
            "concept_quality_summary": concept_quality_summary,
            "blocks": blocks,
        }
        Path(out_path).write_text(json.dumps(report, indent=2))
        print(f"saved: {out_path}", flush=True)

        if args.pyxai_export and seed == seeds[0] and pyxai_payload:
            def to_nodes(t_, nid):
                if t_.children_left[nid] == -1 or t_.feature[nid] < 0:
                    return {"f": -1, "c": None}  # filled by caller with class
                return {"f": int(t_.feature[nid]),
                        "t": float(t_.threshold[nid]),
                        "l": to_nodes(t_, int(t_.children_left[nid])),
                        "r": to_nodes(t_, int(t_.children_right[nid]))}

            def to_nodes_clf(dt_):
                t_ = dt_.tree_
                def rec(nid):
                    if t_.children_left[nid] == -1 or t_.feature[nid] < 0:
                        return {"f": -1, "c": int(
                            dt_.classes_[int(np.argmax(t_.value[nid][0]))])}
                    return {"f": int(t_.feature[nid]),
                            "t": float(t_.threshold[nid]),
                            "l": rec(int(t_.children_left[nid])),
                            "r": rec(int(t_.children_right[nid]))}
                return rec(0)

            payload = {
                "n_features": len(feat_names),
                "feature_names": feat_names,
                "numeric": False,
                "trees": {bn: to_nodes_clf(dt) for bn, dt in tree_store.items()},
                "instances": pyxai_payload,
            }
            Path(args.pyxai_export).write_text(json.dumps(payload))
            print(f"saved: {args.pyxai_export} "
                  f"({len(pyxai_payload)} instances, "
                  f"{len(tree_store)} trees)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
