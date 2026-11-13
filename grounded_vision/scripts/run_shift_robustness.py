"""
Script: run_shift_robustness.py
================================
Round 5 (2026-11-08): does the certified reason survive small
content-preserving image shifts and rescales?

Regions here are FIXED per part (fixed-size template box centered on
the annotated part point -- see build_predicate_rung_table.py notes),
so a content-preserving perturbation moves the image AND the boxes
together; the predicate re-reads the same content.  Metrics per
perturbation on the K=4 subset (118 test images, committed tree):

- prediction stability vs the unperturbed encode;
- reason stability: exact literal set of the native minimal reason;
- all-clauses-annotation-correct rate (the guarantee event);
- the certified-literal mechanism cell: for images whose unperturbed
  reason is the single clause bill_color=yellow|neg, does it remain
  exactly that clause and true?

Sanity gate: re-encoding unperturbed images must reproduce >= 99% of
the cached reader bits (else the transform replication is off and the
run aborts).

PRE-REGISTERED PREDICTIONS (before the run):
  PS1 prediction stability >= 95% at 2% shifts, >= 85% at 5% and at
     scale 0.95/1.05.
  PS2 single-clause (k=1) reasons stay identical >= 95% under 2%
     shifts.
  PS3 reason instability concentrates in multi-clause reasons.
  PS4 the all-correct rate under perturbation stays within 5 points
     of 25.4%.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from torchvision import transforms as T
from torchvision.transforms import functional as VF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader
from grounded_vision.perception.backbone import (FrozenDINOBackbone,
                                                 extract_backbone_features)
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend

IMNET = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    dtr = np.load(Path(args.cache_dir) / "train_features.npz")
    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=args.seed)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=args.seed)
    vocab = ds_train.bound_attributes
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND, attributes=list(vocab))
    feat_names = compiler.feature_names
    attr_by_part = defaultdict(list)
    for j, a in enumerate(vocab):
        attr_by_part[a.part].append(j)
    part_bits = [n for n in feat_names if n.startswith("has_part(")]
    name_to_j = {f"{a.part}:{a.attr_name}={a.value}": j for j, a in enumerate(vocab)}

    # ---- readers (protocol identical to every prior run) --------------------
    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    pooled = {}
    fm_all = torch.from_numpy(dtr["feature_maps"].astype(np.float32))
    for i, row in enumerate(tr_rows):
        sid, _m, boxes, vis, _a = row
        regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                for p in ds_train.part_names if vis.get(p, 0.0) > 0.5 and p in boxes]
        if regs:
            p = pool_region_features(fm_all[i:i + 1], regs)
            pooled[int(sid)] = {r.part_name: p[r.id].numpy() for r in regs}
    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    sids = np.array(sorted(pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(args.val_fraction * len(sids))
    fit_sids = set(sids[perm[n_val:]].tolist())
    reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
    region_rows = defaultdict(list)
    for sid in fit_sids:
        for part, vec in pooled[sid].items():
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
    print("--> readers trained", flush=True)

    def vec_from_probs(parts):
        d = {}
        for part, vec in parts.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                with torch.no_grad():
                    t = torch.from_numpy(vec.reshape(1, -1)).float()
                    p = float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
                d[f"{a.part}:{a.attr_name}={a.value}"] = 1 if p >= 0.5 else 0
        for n in part_bits:
            d[n] = 1
        return np.array([d.get(n, 0) for n in feat_names], dtype=np.int64)

    # K=4 tree from cached train bits
    sid_to_label_tr = {int(s): int(l) for s, l in zip(dtr["sample_ids"], dtr["labels"])}
    tr_cp_vec = {}
    for sid in sorted(fit_sids):
        tr_cp_vec[sid] = vec_from_probs(pooled[sid])
    tr_ids = [s for s in sorted(fit_sids) if sid_to_label_tr[s] < 4]
    X_fit = np.stack([tr_cp_vec[s] for s in tr_ids])
    y_fit = np.array([sid_to_label_tr[s] for s in tr_ids])
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                random_state=42).fit(X_fit, y_fit)
    head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                               feature_names=feat_names)
    head.clf, head.is_fitted = dt, True
    backend = PyXAIBackend(head)

    backbone = FrozenDINOBackbone()
    path_by_imgid = {}
    for line in open(Path(args.data_dir) / "images.txt"):
        p = line.split()
        path_by_imgid[int(p[0])] = Path(args.data_dir) / "images" / p[1]

    # perturbations: (name, translate_px, scale)
    perturbations = []
    for pct in (2, 5):
        px = int(round(224 * pct / 100))
        for (dx, dy, tag) in ((px, 0, "right"), (-px, 0, "left"),
                              (0, px, "down"), (0, -px, "up")):
            perturbations.append((f"shift{pct}%_{tag}", (dx, dy), 1.0))
    for s in (0.95, 1.05):
        perturbations.append((f"scale{s}", (0, 0), s))

    def encode(pil224):
        x = IMNET(T.functional.to_tensor(pil224)).unsqueeze(0).to(backbone.device)
        with torch.no_grad():
            feats = extract_backbone_features(backbone, x)
        return feats["feature_map"][0].detach().cpu()

    def shifted_boxes(boxes, dx, dy, s):
        out = {}
        for p, (y0, x0, y1, x1) in boxes.items():
            ny0, nx0 = (y0 - 0.5) * s + 0.5 + dy / 224, (x0 - 0.5) * s + 0.5 + dx / 224
            ny1, nx1 = (y1 - 0.5) * s + 0.5 + dy / 224, (x1 - 0.5) * s + 0.5 + dx / 224
            out[p] = (min(max(ny0, 0), 1), min(max(nx0, 0), 1),
                      min(max(ny1, 0), 1), min(max(nx1, 0), 1))
        return out

    results = {name: {"pred_same": 0, "reason_same": 0, "allc": 0, "n": 0,
                      "k1_reason_same": 0, "n_k1": 0, "kgt_reason_same": 0,
                      "n_kgt": 0} for name, _, _ in perturbations}
    mechanism = {name: {"still_yellow_neg_k1": 0, "still_true": 0, "n": 0}
                 for name, _, _ in perturbations}
    sanity_same = sanity_n = 0
    cache_bits = None

    dte = np.load(Path(args.cache_dir) / "test_features.npz")
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}
    te_rows = [r for r in meta_rows(ds_test, dte["sample_ids"]) if sid_to_label_te[r[0]] < 4]
    te_fm = dte["feature_maps"]
    te_order = {int(s): i for i, s in enumerate(dte["sample_ids"])}

    base_rows = []
    for k, (sid, meta_sid, boxes, vis, attrs) in enumerate(te_rows):
        base_rows.append((sid, meta_sid, boxes, vis, attrs))

    print(f"--> {len(base_rows)} K=4 test images; encoding", flush=True)
    for i, (sid, meta_sid, boxes, vis, attrs) in enumerate(base_rows):
        pil = T.Resize((224, 224))(Image.open(path_by_imgid[int(meta_sid)]).convert("RGB"))
        # ground truth clause bits (annotation)
        gtb = {f"{a.part}:{a.attr_name}={a.value}": int(attrs[j])
               for j, a in enumerate(vocab)}
        for n in part_bits:
            p = n[len("has_part("):-1]
            gtb[n] = 1 if vis.get(p, 0.0) > 0.5 else 0

        # --- unperturbed: fresh encode (sanity vs cache) + cached encode
        fmap_cache = torch.from_numpy(
            te_fm[te_order[sid]].astype(np.float32))
        parts_cache = {}
        for p in ds_train.part_names:
            if vis.get(p, 0.0) > 0.5 and p in boxes:
                r = Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                pp = pool_region_features(fmap_cache.unsqueeze(0), [r])
                parts_cache[p] = pp[r.id].numpy()
        v_cache = vec_from_probs(parts_cache)
        fmap_fresh = encode(pil)
        parts_fresh = {}
        for p in parts_cache:
            r = Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
            pp = pool_region_features(fmap_fresh.unsqueeze(0), [r])
            parts_fresh[p] = pp[r.id].numpy()
        v_fresh = vec_from_probs(parts_fresh)
        sanity_same += int((v_cache == v_fresh).all()); sanity_n += 1

        # baseline for perturbation deltas = the fresh unperturbed encode
        # (within-pipeline comparison; cache-vs-fresh is the sanity check)
        pred0 = int(dt.predict(v_fresh.reshape(1, -1))[0])
        exp0 = backend.explain_sufficient(v_fresh, pred0, sample_id=sid)
        lits0 = tuple(sorted((c.feature_idx, c.operator) for c in exp0.conditions))
        allc0 = all(gtb.get(feat_names[c.feature_idx]) == (1 if c.operator != "<=" else 0)
                    for c in exp0.conditions)
        k1 = len(exp0.conditions) == 1

        for name, (dx, dy), s in perturbations:
            pil_p = VF.affine(pil, angle=0, translate=(dx, dy), scale=s, shear=0)
            fmap_p = encode(pil_p)
            boxes_p = shifted_boxes(boxes, dx, dy, s)
            parts_p = {}
            for p in ds_train.part_names:
                if vis.get(p, 0.0) > 0.5 and p in boxes:
                    r = Region(id=f"region_{p}", part_name=p, bbox=boxes_p[p])
                    pp = pool_region_features(fmap_p.unsqueeze(0), [r])
                    parts_p[p] = pp[r.id].numpy()
            vp = vec_from_probs(parts_p)
            predp = int(dt.predict(vp.reshape(1, -1))[0])
            expp = backend.explain_sufficient(vp, predp, sample_id=sid)
            litsp = tuple(sorted((c.feature_idx, c.operator) for c in expp.conditions))
            allcp = all(gtb.get(feat_names[c.feature_idx]) == (1 if c.operator != "<=" else 0)
                        for c in expp.conditions)
            r_ = results[name]
            r_["n"] += 1
            r_["pred_same"] += int(predp == pred0)
            r_["reason_same"] += int(litsp == lits0)
            r_["allc"] += int(allcp)
            if k1:
                r_["n_k1"] += 1
                r_["k1_reason_same"] += int(litsp == lits0)
            if len(exp0.conditions) > 1:
                r_["n_kgt"] += 1
                r_["kgt_reason_same"] += int(litsp == lits0)
            # mechanism cell: unperturbed reason == single yellow|neg clause
            if (k1 and len(exp0.conditions) == 1
                    and feat_names[exp0.conditions[0].feature_idx]
                    == "beak:bill_color=yellow"
                    and exp0.conditions[0].operator == "<="):
                m = mechanism[name]
                m["n"] += 1
                m["still_yellow_neg_k1"] += int(
                    len(expp.conditions) == 1
                    and feat_names[expp.conditions[0].feature_idx]
                    == "beak:bill_color=yellow"
                    and expp.conditions[0].operator == "<=")
                m["still_true"] += int(allcp)

    sanity_rate = sanity_same / sanity_n
    print(f"--> SANITY fresh-encode bit match vs cache: {sanity_rate:.3f}", flush=True)
    out = {
        "protocol": {
            "question": "Does the certified minimal reason survive small "
                        "content-preserving shifts/rescales (image AND fixed "
                        "part-box regions moved together)?",
            "predictions_pre_registered": [
                "PS1 pred stability >= 95% @2%, >= 85% @5% and scale",
                "PS2 k=1 reasons identical >= 95% @2%",
                "PS3 instability concentrates in multi-clause reasons",
                "PS4 all-correct within 5 points of 25.4%",
            ],
            "substrate": "K=4 committed tree (depth 12 leaf 4 seed 42), "
                         "readers seed-42 protocol, GT fixed part boxes, "
                         "dinov2_vits14 224x224 eval transform",
            "sanity_bit_match_vs_cache": sanity_rate,
        },
        "results": results,
        "mechanism_yellow_neg": mechanism,
    }
    if sanity_rate < 0.99:
        out["protocol"]["ABORTED"] = ("fresh-encode replication below 99%; "
                                      "numbers below are unreliable")
    Path(args.out).write_text(json.dumps(out, indent=2))
    for name, _, _ in perturbations:
        r_ = results[name]
        print(f"  {name:14s} pred_same {r_['pred_same']/r_['n']:.3f} "
              f"reason_same {r_['reason_same']/r_['n']:.3f} "
              f"allc {r_['allc']/r_['n']:.3f} "
              f"k1_same {r_['k1_reason_same']}/{r_['n_k1']} "
              f"k>1_same {r_['kgt_reason_same']}/{r_['n_kgt']}", flush=True)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
