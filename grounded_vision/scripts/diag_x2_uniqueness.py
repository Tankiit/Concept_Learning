"""Diagnostic (2026-11-08): why does UAR == ORD on every instance in
x2_uncertainty_reasons.json? Rebuilds the frozen Setting T K=4 pipeline
(seed 42, identical draws) and enumerates ALL sufficient subsets of the
path literals per test instance, reporting:
  - count of sufficient subsets and of inclusion-minimal ones
  - min cardinality and how many sets achieve it
  - leave-one-out necessity of every ORD literal
  - among minimal subsets with != ORD, the best available q vs q(ORD)
"""
from __future__ import annotations
import os, sys, math
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
from run_x2_uncertainty import leaf_bounds, literal_contradicts_leaf, uar_from_path

CACHE = "/Users/tanmoy/research/data/grounded_vision_runs/cub_local_et"
DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
SEED, K = 42, 4

dtr = np.load(Path(CACHE) / "train_features.npz")
dte = np.load(Path(CACHE) / "test_features.npz")
ds_train = CUBDataset(data_dir=DATA, split="train", use_mock_if_missing=False, seed=42)
ds_test = CUBDataset(data_dir=DATA, split="test", use_mock_if_missing=False, seed=42)
vocab = ds_train.bound_attributes
compiler = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=list(vocab))
feat_names = compiler.feature_names
attr_by_part = defaultdict(list)
for j, a in enumerate(vocab):
    attr_by_part[a.part].append(j)
part_feat_idx = [compiler.feature_to_idx[n] for n in feat_names if n.startswith("has_part(")]

def meta_rows(ds, ids):
    return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

tr_rows, te_rows = meta_rows(ds_train, dtr["sample_ids"]), meta_rows(ds_test, dte["sample_ids"])

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

print("--> pooling", flush=True)
tr_pooled, te_pooled = pooled(tr_rows, dtr["feature_maps"]), pooled(te_rows, dte["feature_maps"])
tr_gt = {int(m[0]): m[4] for m in tr_rows}
sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
sids = np.array(sorted(tr_pooled.keys()))
perm = rng.permutation(len(sids))
fit_sids = set(sids[perm[int(0.15 * len(sids)):]].tolist())
reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
opt = torch.optim.Adam(reader.parameters(), lr=1e-3)
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
    y = torch.from_numpy(np.array([tr_gt[s][cols] for s, _ in rows], dtype=np.float32))
    pw = ((y.shape[0] - y.sum(dim=0)) / y.sum(dim=0).clamp(min=1.0)).clamp(max=50.0)
    region_data[part] = (X, y, pw, cols)
for _ in range(40):
    loss = 0.0
    for X, y, pw, cols in region_data.values():
        W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
        b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
        loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
    opt.zero_grad(); loss.backward(); opt.step()
print("--> readers trained", flush=True)

def scores_map(parts):
    d = {}
    for part, vec in parts.items():
        for j in attr_by_part.get(part, []):
            a = vocab[j]
            with torch.no_grad():
                t = torch.from_numpy(vec.reshape(1, -1)).float()
                d[f"{a.part}:{a.attr_name}={a.value}"] = float(
                    torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
    return d

def vecT(parts):
    d = scores_map(parts)
    sc = np.zeros(len(feat_names))
    for name, p in d.items():
        sc[compiler.feature_to_idx[name]] = p
    b = (sc >= 0.5).astype(np.int64)
    for fi in part_feat_idx:
        b[fi] = 1
    return b, sc

tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < K]
te_ids = [s for s in te_pooled.keys() if sid_to_label_te[s] < K]
X_tr = np.stack([vecT(tr_pooled[s])[0] for s in tr_ids])
y_tr = np.array([sid_to_label_tr[s] for s in tr_ids])
dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4, random_state=42).fit(X_tr, y_tr)
leaves = leaf_bounds(dt)
print(f"--> tree: {len(leaves)} leaves, depth {dt.get_depth()}", flush=True)

from grounded_vision.reasoning.tree_head import CertifiableTreeHead
head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4, feature_names=feat_names)
head.clf, head.is_fitted = dt, True
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
back = PyXAIBackend(head)

n_minimal_1, n_minimal_multi, n_suff_hist = 0, 0, defaultdict(int)
loo_all_nec, n_multi_with_q_alt = 0, 0
examples = []
for i, s in enumerate(te_ids):
    v, sc = vecT(te_pooled[s])
    c = int(dt.predict(v.reshape(1, -1))[0])
    exp = back.explain_sufficient(v, c, sample_id=s)
    ord_set = frozenset((int(cc.feature_idx), cc.operator != "<=") for cc in exp.conditions)
    steps = head.get_decision_path(v)
    path_lits = [(st.feature_idx, st.is_left, st.threshold) for st in steps]
    confs = [sc[f] if not is_left else 1.0 - sc[f] for f, is_left, _t in path_lits]
    leaf_pred = c
    neg_masks = []
    for lf in leaves:
        if lf["pred"] == leaf_pred:
            continue
        mask = 0
        for k, (fidx, is_left, thr) in enumerate(path_lits):
            if literal_contradicts_leaf(fidx, is_left, thr, lf):
                mask |= (1 << k)
        neg_masks.append(mask)
    m = len(path_lits)
    suff = []
    for ss in range(1, 1 << m):
        if all((nm & ss) for nm in neg_masks):
            suff.append(ss)
    n_suff_hist[len(suff)] += 1
    minimal = [ss for ss in suff
               if not any((ss2 & ss) == ss2 for ss2 in suff if ss2 != ss)]
    if len(minimal) == 1:
        n_minimal_1 += 1
    else:
        n_minimal_multi += 1
        q_of = lambda ss: math.exp(sum(-math.log(max(confs[k], 1e-12))
                                       for k in range(m) if ss >> k & 1))
        alts = [(ss, q_of(ss)) for ss in minimal]
        best_alt = max(alts, key=lambda t: (t[1], -bin(t[0]).count("1")))
        if best_alt[0] != minimal[0]:
            n_multi_with_q_alt += 1
        if len(examples) < 5:
            examples.append({"sid": int(s), "n_minimal": len(minimal),
                             "q_best": round(best_alt[1], 4),
                             "q_first": round(q_of(minimal[0]), 4)})
    # LOO necessity of ORD literals: recompute blocking without literal
    ord_pos = {k for k, (fidx, is_left, _t) in enumerate(path_lits)
               if (fidx, not is_left) in ord_set}
    all_nec = True
    for k in ord_pos:
        ss_without = 0
        for kk in ord_pos:
            if kk != k:
                ss_without |= (1 << kk)
        if all((nm & ss_without) for nm in neg_masks):
            all_nec = False
            break
    loo_all_nec += int(all_nec)

print(f"instances: {len(te_ids)}")
print(f"unique minimal sufficient subset: {n_minimal_1} | multiple: {n_minimal_multi} "
      f"| of multiple, argmax-q != first: {n_multi_with_q_alt}")
print(f"every ORD literal leave-one-out necessary: {loo_all_nec}/{len(te_ids)}")
print("sufficient-subset count histogram (top):",
      dict(sorted(n_suff_hist.items(), key=lambda t: -t[1])[:6]))
print("multi-minimal examples:", examples)
