"""
Script: build_predicate_rung_table.py
======================================
Round 4 (2026-11-08): the predicate x rung table for the review's
claim ladder.  Every predicate p_j = [reader_j(region_j(x)) >= 0.5]
over a FIXED annotated part box (GT-region protocol).

Rungs (each needs more evidence than the one below):
  1 operational   - the bit the tree sees (free, exact)
  2 localized     - perturbation specificity CI-lower > 1 under ALL
                    THREE control families (milestone v2; reported for
                    both deployed-reader and probe instruments)
  3 aligned       - some (concept, polarity) literal certified on the
                    calibration split at tau=0.8 (Wilson-lower; raw and
                    Bonferroni), from the frontier run
  4 semantic      - no EXCESS class information: NMI(bit; class) -
                    NMI(annotation_bit; class) <= eps on the full test
                    split (one-sided; the annotation bit's class
                    information is the concept's legitimate share).
                    "Concept" status = rung2 AND rung3 AND rung4.

PRE-REGISTERED PREDICTIONS (before the run):
  PR1: >= 20 predicates reach rung 3 via a NEGATIVE literal at tau=0.8;
       zero via a positive literal.
  PR2: bill_color=yellow PASSES the one-sided excess test (its class
       information does not exceed the annotation's) despite its
       0.11 positive-literal precision -- alignment and leakage are
       independent axes.
  PR3: <= 4 predicates achieve full "concept" status (rung2-reader
       AND rung3 AND rung4).
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.data.cub import CUBDataset
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.perception.readers import AttributeReader
from grounded_vision.reasoning.features import FeatureCompiler, RepresentationMode


def nmi(x, y):
    n = len(x)
    cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    mi = sum(v / n * math.log2((v / n) / (cx[a] / n * cy[b] / n))
             for (a, b), v in cxy.items())
    hy = -sum(v / n * math.log2(v / n) for v in cy.values())
    return mi / hy if hy > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--milestone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # ---- rung 2 from the milestone, rung 3 from the frontier ----------------
    mile = json.load(open(args.milestone))
    rung2 = {}
    for p in mile["predicates"]:
        all3 = all(p["passes_by_control"].values())
        rung2.setdefault(p["feature"], {})[p["instrument"]] = all3
    front = json.load(open(args.frontier))
    cert = front["literal_certification"]
    rung3 = {}
    for key, v in cert.items():
        if v["certified_at"].get(str(args.tau)):
            name = key.rsplit("|", 1)[0]
            rung3.setdefault(name, []).append(key.rsplit("|", 1)[1])

    dtr = np.load(Path(args.cache_dir) / "train_features.npz")
    dte = np.load(Path(args.cache_dir) / "test_features.npz")
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

    def meta_rows(ds, ids):
        return [(int(ids[k]), *ds.meta(k)) for k in range(len(ids))]

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])
    print(f"--> pooling train {len(tr_rows)} / test {len(te_rows)}", flush=True)

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

    tr_pooled = pooled(tr_rows, dtr["feature_maps"])
    te_pooled = pooled(te_rows, dte["feature_maps"])
    tr_gt = {int(m[0]): m[4] for m in tr_rows}
    te_gt = {int(m[0]): m[4] for m in te_rows}
    te_vis = {int(m[0]): m[3] for m in te_rows}
    sid_to_label_te = {int(s): int(l) for s, l in zip(dte["sample_ids"], dte["labels"])}

    sids = np.array(sorted(tr_pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(args.val_fraction * len(sids))
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
    print("--> readers trained; scoring full test split", flush=True)

    bits = {n: [] for n in feat_names}
    gts = {n: [] for n in feat_names}
    labels = []
    for sid in sorted(te_pooled.keys()):
        cp = {}
        for part, vec in te_pooled[sid].items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                with torch.no_grad():
                    t = torch.from_numpy(vec.reshape(1, -1)).float()
                    p = float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])
                cp[f"{a.part}:{a.attr_name}={a.value}"] = 1 if p >= 0.5 else 0
        for n in part_bits:
            cp[n] = 1
        gt = {f"{a.part}:{a.attr_name}={a.value}": int(te_gt[sid][j])
              for j, a in enumerate(vocab)}
        for n in part_bits:
            p = n[len("has_part("):-1]
            gt[n] = 1 if te_vis[sid].get(p, 0.0) > 0.5 else 0
        for n in feat_names:
            bits[n].append(cp.get(n, 0))
            gts[n].append(gt.get(n, 0))
        labels.append(sid_to_label_te[sid])

    table = []
    for n in feat_names:
        nmi_bit = nmi(bits[n], labels)
        nmi_gt = nmi(gts[n], labels)
        excess = nmi_bit - nmi_gt
        r2 = rung2.get(n, {})
        r3 = rung3.get(n, [])
        r4 = excess <= args.eps
        r2_reader = r2.get("reader", False)
        r2_probe = r2.get("probe", False)
        rung = 1
        if r2_reader or r2_probe:
            rung = 2
        if r3:
            rung = 3
        concept = r2_reader and r3 and r4
        table.append({
            "predicate": n, "rung_reached": rung,
            "rung2_localized_reader": r2_reader,
            "rung2_localized_probe": r2_probe,
            "rung3_aligned_literals_tau0.8": r3,
            "rung4_no_excess_leakage": r4,
            "nmi_bit_class": round(nmi_bit, 3),
            "nmi_annotation_class": round(nmi_gt, 3),
            "excess_leakage": round(excess, 3),
            "concept_status": concept,
        })
    n_r1 = sum(1 for t in table if t["rung_reached"] == 1)
    n_r2 = sum(1 for t in table if t["rung_reached"] == 2)
    n_r3 = sum(1 for t in table if t["rung_reached"] == 3)
    n_concept = sum(1 for t in table if t["concept_status"])
    print(f"--> rung counts: R1 {n_r1} | R2 {n_r2} | R3 {n_r3} | "
          f"full concept (R2r+R3+R4) {n_concept}", flush=True)
    y = next(t for t in table if t["predicate"] == "beak:bill_color=yellow")
    print(f"--> yellow: {y}", flush=True)

    report = {
        "protocol": {
            "ladder": ("1 operational (bit the tree sees); 2 localized "
                       "(specificity CI-lower>1 under all 3 control families, "
                       "milestone v2); 3 aligned (literal certified on "
                       "calibration split, tau=0.8 Wilson-lower, frontier run); "
                       "4 semantic (one-sided excess class information "
                       "NMI(bit;class)-NMI(annot;class) <= eps on full test)"),
            "predictions_pre_registered": [
                "PR1 >= 20 predicates reach rung 3 via a negative literal; 0 via positive",
                "PR2 bill_color=yellow passes one-sided excess despite 0.11 pos precision",
                "PR3 <= 4 predicates reach full concept status",
            ],
            "theta": "0.5 fixed a priori for all readers (not tuned; no split leakage)",
            "region": "fixed annotated part boxes (GT-region protocol)",
            "absence_semantics": ("binary encoding: absent = reader-below-0.5 "
                                  "OR part-unobserved (unknown->0); tristate "
                                  "rerun pending"),
            "eps": args.eps, "tau": args.tau, "n_test": len(labels),
        },
        "table": table,
        "counts": {"rung1_only": n_r1, "rung2": n_r2, "rung3": n_r3,
                   "concept": n_concept},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
