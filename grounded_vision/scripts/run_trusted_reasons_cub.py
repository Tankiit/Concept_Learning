"""
Script: run_trusted_reasons_cub.py
==================================
Upgrades review "sentence 3" from a lemma to a measured result, and computes
E1, on real CUB under the official split.

Substrate (all local, eval-transform caches):
- AttributeReader heads trained per the fit_perception protocol on a TRAIN
  sub-split (85%); a held-out 15% VALIDATION sub-split is used ONLY to
  measure per-concept reliability (independent evidence for the trust set).
- Concept tree: CART (depth 12, leaf 4, seed 42) on predicted 36-bit BOUND
  concept vectors (canonical 28 attributes + 8 part bits) of train images.

E1 (local): P(tree(predicted concepts) == tree(GT concepts)) on test images,
reported against P(all clauses of the default reason GT-correct) - i.e., does
"the reason held" track "the output held".

Trusted-only table (payload for scripts/run_pyxai_modal.py):
- trust set = concepts whose validation agreement >= delta (default 0.75);
  excluded set = the complement (untrusted concepts), constant per instance.
- Modal returns coverage (non-abstaining reasons), their literals, and sizes;
  --evaluate then computes P(all clauses GT-correct | covered) and compares
  with the independence prediction at the same composition.

Usage:
  build:  python scripts/run_trusted_reasons_cub.py build \
              --cache-dir ... --data-dir ... --out-dir ...
  evaluate: python scripts/run_trusted_reasons_cub.py evaluate \
              --build-dir ... --arm-results <pyxai json> --out <json>
"""

from __future__ import annotations
import argparse
import json
import os
import sys
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


def pooled_by_part(fmaps, metas, part_names):
    """fmaps: [N,384,16,16]; metas: list of (sid, vboxes_all, vis). Returns
    {sid: {part: np[384]}} for VISIBLE parts, using the framework pooling."""
    out = {}
    fm = torch.from_numpy(fmaps.astype(np.float32))
    for i, m in enumerate(metas):
        sid, boxes, vis = m[0], m[1], m[2]
        regs = [Region(id=f"region_{p}", part_name=p, bbox=boxes[p])
                for p in part_names if vis.get(p, 0.0) > 0.5 and p in boxes]
        if not regs:
            continue
        pooled = pool_region_features(fm[i:i + 1], regs)
        out[int(sid)] = {r.part_name: pooled[r.id].numpy() for r in regs}
    return out


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--delta", type=float, default=0.75,
                    help="primary trust threshold (table row)")
    ap.add_argument("--delta-grid", default="0.60,0.65,0.70,0.75",
                    help="comma-separated thresholds; one payload each")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(sys.argv[2:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    dtr = np.load(Path(args.cache_dir) / "train_features.npz")
    dte = np.load(Path(args.cache_dir) / "test_features.npz")

    ds_train = CUBDataset(data_dir=args.data_dir, split="train",
                          use_mock_if_missing=False, seed=args.seed)
    ds_test = CUBDataset(data_dir=args.data_dir, split="test",
                         use_mock_if_missing=False, seed=args.seed)
    vocab = ds_train.bound_attributes
    compiler = FeatureCompiler(mode=RepresentationMode.BOUND,
                                attributes=list(vocab))
    feat_names = compiler.feature_names  # 36 sorted BOUND names
    fidx = compiler.feature_to_idx
    n_features = len(feat_names)
    attr_by_part = {}
    for j, a in enumerate(vocab):
        attr_by_part.setdefault(a.part, []).append(j)
    part_bits = [n for n in feat_names if n.startswith("has_part(")]
    print(f"--> vocabulary: {len(vocab)} attributes -> {n_features} features",
          flush=True)

    def metas_of(ds, ids):
        return [ds.meta(int(i)) + (int(ids[k]),) for k, i in
                (enumerate(range(len(ids))))] if False else None

    # metadata aligned to cache rows (cache preserves dataset order)
    def meta_rows(ds, cache_ids):
        rows = []
        for k in range(len(cache_ids)):
            sid, boxes, vis, attrs = ds.meta(k)
            rows.append((int(cache_ids[k]), boxes, vis, attrs))
        return rows

    tr_rows = meta_rows(ds_train, dtr["sample_ids"])
    te_rows = meta_rows(ds_test, dte["sample_ids"])
    print(f"--> pooling features: train {len(tr_rows)} / test {len(te_rows)}",
          flush=True)
    tr_pooled = pooled_by_part(dtr["feature_maps"], tr_rows, ds_train.part_names)
    te_pooled = pooled_by_part(dte["feature_maps"], te_rows, ds_test.part_names)

    # GT labels aligned per sid (canonical 28 bits)
    tr_gt = {int(m[0]): m[3] for m in tr_rows}
    te_gt = {int(m[0]): m[3] for m in te_rows}
    tr_vis = {int(m[0]): m[2] for m in tr_rows}
    te_vis = {int(m[0]): m[2] for m in te_rows}

    # ---- train/validation sub-split (by image) ------------------------------
    sids = np.array(sorted(tr_pooled.keys()))
    perm = rng.permutation(len(sids))
    n_val = int(args.val_fraction * len(sids))
    val_sids = set(sids[perm[:n_val]].tolist())
    fit_sids = set(sids[perm[n_val:]].tolist())

    # ---- reader training (fit_perception protocol, train sub-split) ---------
    region_rows = {}
    for sid in fit_sids:
        for part, vec in tr_pooled[sid].items():
            region_rows.setdefault(part, []).append((sid, vec))
    reader = AttributeReader(embedding_dim=384, attributes=list(vocab))
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
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
    for epoch in range(args.epochs):
        loss = 0.0
        for part, (X, y, pw, cols) in region_data.items():
            W = torch.cat([reader.classifiers[f"attr_{j}"].weight for j in cols], dim=0)
            b = torch.cat([reader.classifiers[f"attr_{j}"].bias for j in cols], dim=0)
            loss = loss + F.binary_cross_entropy_with_logits(X @ W.T + b, y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"--> reader trained on {len(fit_sids)} images "
          f"({len(val_sids)} held out for reliability)", flush=True)

    def p_attr(j, vec):
        with torch.no_grad():
            t = torch.from_numpy(vec.reshape(1, -1)).float()
            return float(torch.sigmoid(reader.classifiers[f"attr_{j}"](t))[0, 0])

    def concept_vector(pooled, vis):
        """36-bit BOUND vector: attr bit = visible & prob>=0.5; part bit = visible."""
        bits = {}
        for part, vec in pooled.items():
            for j in attr_by_part.get(part, []):
                a = vocab[j]
                bits[f"{a.part}:{a.attr_name}={a.value}"] = \
                    1.0 if p_attr(j, vec) >= 0.5 else 0.0
        for n in part_bits:
            p = n[len("has_part("):-1]
            bits[n] = 1.0 if vis.get(p, 0.0) > 0.5 else 0.0
        return np.array([bits.get(n, 0.0) for n in feat_names], dtype=np.int64)

    def gt_vector(gt, vis):
        bits = {}
        for j, a in enumerate(vocab):
            bits[f"{a.part}:{a.attr_name}={a.value}"] = float(gt[j])
        for n in part_bits:
            p = n[len("has_part("):-1]
            bits[n] = 1.0 if vis.get(p, 0.0) > 0.5 else 0.0
        return np.array([bits.get(n, 0.0) for n in feat_names], dtype=np.int64)

    X_pred = {sid: concept_vector(tr_pooled[sid], tr_vis[sid]) for sid in sids}
    # ---- per-concept validation reliability (independent evidence) ----------
    val_acc = {}
    for j, a in enumerate(vocab):
        name = f"{a.part}:{a.attr_name}={a.value}"
        ok, n = 0, 0
        for sid in val_sids:
            if a.part in tr_pooled[sid]:
                n += 1
                ok += int(X_pred[sid][fidx[name]] == tr_gt[sid][j])
        val_acc[name] = {"accuracy": ok / n if n else None, "n": n}

    # ---- concept tree on predicted concepts (train sub-split) ---------------
    X_fit = np.stack([X_pred[s] for s in sorted(fit_sids)])
    label_by_sid = {int(m[0]): None for m in tr_rows}
    labels_cache = dtr["labels"]
    sid_to_label = {int(s): int(l) for s, l in zip(dtr["sample_ids"], labels_cache)}
    y_fit = np.array([sid_to_label[s] for s in sorted(fit_sids)])
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                random_state=42).fit(X_fit, y_fit)
    head = CertifiableTreeHead(max_depth=12, min_samples_leaf=4,
                               feature_names=feat_names)
    head.clf, head.is_fitted = dt, True
    backend = PyXAIBackend(head)

    # ---- test subsample: default reasons, E1, clause correctness -----------
    te_sids = np.array(sorted(te_pooled.keys()))
    sel = np.sort(np.random.RandomState(7).permutation(len(te_sids))[:args.n_test])
    te_sel = [int(te_sids[i]) for i in sel]

    def clause_ok(lit, gt_vec):
        f = abs(lit) - 1
        return bool(gt_vec[f] == (1 if lit > 0 else 0))

    rows_out = []
    n_out_match = 0
    for sid in te_sel:
        xp = concept_vector(te_pooled[sid], te_vis[sid])
        xg = gt_vector(te_gt[sid], te_vis[sid])
        pred = int(dt.predict(xp.reshape(1, -1))[0])
        out_match = int(dt.predict(xg.reshape(1, -1))[0]) == pred
        n_out_match += out_match
        exp = backend.explain_sufficient(xp, pred, sample_id=sid)
        lits = [(-1 if c.operator == "<=" else 1) * (c.feature_idx + 1)
                for c in exp.conditions]
        rows_out.append({
            "sample_id": sid, "prediction": pred,
            "output_match_gt": bool(out_match),
            "x": xp.tolist(),
            "x_gt": xg.tolist(),
            "native_reason_literals": lits,
            "native_reason_size": len(lits),
            "all_clauses_correct": bool(all(clause_ok(l, xg) for l in lits)),
            "present_clauses": sum(1 for l in lits if l > 0),
            "present_clauses_correct": sum(
                1 for l in lits if l > 0 and clause_ok(l, xg)),
        })
    p_all = float(np.mean([r["all_clauses_correct"] for r in rows_out]))
    e1 = n_out_match / len(rows_out)
    mean_size = float(np.mean([r["native_reason_size"] for r in rows_out]))
    present_c, present_t = 0, 0
    for r in rows_out:
        present_c += r["present_clauses_correct"]
        present_t += r["present_clauses"]
    print(f"--> E1 output-match: {e1:.3f} | P(all clauses correct): {p_all:.3f} "
          f"| mean reason {mean_size:.2f} | present-clause precision "
          f"{present_c / max(1, present_t):.3f}", flush=True)

    # independence prediction: product of per-clause correctness rates
    clause_acc = {}
    for name, d in val_acc.items():
        if d["accuracy"] is not None:
            clause_acc[fidx[name]] = d["accuracy"]
    ind_pred = []
    for r in rows_out:
        ps = [clause_acc.get(abs(l) - 1, 0.5) for l in r["native_reason_literals"]]
        ind_pred.append(float(np.prod(ps)) if ps else 1.0)
    ind_pred = float(np.mean(ind_pred))

    # part-presence bits: predicted from the same visibility annotations as
    # the GT bits (GT-region protocol) -> measure them like any other concept
    for n in part_bits:
        p = n[len("has_part("):-1]
        ok_v, n_v = 0, 0
        for sid in val_sids:
            if p in tr_pooled[sid]:
                n_v += 1
                ok_v += int((tr_vis[sid].get(p, 0.0) > 0.5) ==
                            (tr_vis[sid].get(p, 0.0) > 0.5))
        val_acc[n] = {"accuracy": ok_v / n_v if n_v else None, "n": n_v}

    # ---- export the sklearn tree once (indexed by nid), then one payload per delta
    nodes = [None] * dt.tree_.node_count
    t = dt.tree_
    def walk(nid):
        if t.children_left[nid] == -1:
            nodes[nid] = {"f": -1, "t": 0.0, "l": -1, "r": -1,
                          "c": int(dt.classes_[int(np.argmax(t.value[nid][0]))])}
            return
        nodes[nid] = {"f": int(t.feature[nid]), "t": float(t.threshold[nid]),
                      "l": int(t.children_left[nid]), "r": int(t.children_right[nid])}
        walk(t.children_left[nid]); walk(t.children_right[nid])
    walk(0)

    validated = {n: d["accuracy"] for n, d in val_acc.items()
                 if d["accuracy"] is not None}
    payloads = {}
    trust_report = {}
    for delta in [float(x) for x in args.delta_grid.split(",")]:
        trusted = sorted(n for n, acc in validated.items() if acc >= delta)
        trusted_set = set(trusted)
        untrusted_idx = sorted(fidx[n] for n, acc in validated.items()
                               if n not in trusted_set)
        trust_report[delta] = {"n_trusted": len(trusted),
                               "trusted": trusted,
                               "n_excluded": len(untrusted_idx),
                               "untrusted_feature_indices": untrusted_idx}
        payloads[delta] = {
            "n_features": n_features, "n_classes": int(len(dt.classes_)),
            "numeric": False, "nodes": nodes, "feature_names": feat_names,
            "arm": f"trusted-only delta={delta}; excluded=untrusted concepts",
            "count_sufficient_reasons": {"n_instances": 0, "time_limit": 10},
            "instances": [
                {"sample_id": r["sample_id"], "x": r["x"],
                 "prediction": r["prediction"],
                 "native_reason_literals": r["native_reason_literals"],
                 "native_reason_size": r["native_reason_size"],
                 "excluded_features": untrusted_idx}
                for r in rows_out],
        }
        print(f"--> delta={delta}: trusted {len(trusted)}/{len(validated)} "
              f"| excluded per instance: {len(untrusted_idx)}", flush=True)
    # primary delta = args.delta for the headline report
    trusted = trust_report[args.delta]["trusted"]
    untrusted_idx = trust_report[args.delta]["untrusted_feature_indices"]

    # ---- write one payload per delta -----------------------------------------
    payload_paths = {}
    for delta, payload in payloads.items():
        p = out_dir / f"trusted_reasons_payload_d{delta:.2f}.json"
        p.write_text(json.dumps(payload))
        payload_paths[delta] = str(p)
        print(f"saved: {p}")

    build_summary = {
        "protocol": {
            "cache": "eval-transform (no flip)", "delta": args.delta,
            "n_test": len(rows_out),
            "reader": f"AttributeReader, {args.epochs} epochs, lr {args.lr}",
            "tree": "CART depth 12, leaf 4, seed 42, 200 classes",
            "trust_evidence": "per-concept agreement on the 15% validation sub-split",
        },
        "feature_names": feat_names,
        "validation_reliability": val_acc,
        "trusted_concepts": trusted,
        "untrusted_feature_indices": untrusted_idx,
        "default_row": {
            "coverage": 1.0,
            "p_all_clauses_correct": p_all,
            "independence_prediction": ind_pred,
            "mean_reason_size": mean_size,
            "present_clause_precision": present_c / max(1, present_t),
        },
        "e1_output_match": e1,
        "trust_report_by_delta": {str(k): v for k, v in trust_report.items()},
        "payload_paths": {str(k): v for k, v in payload_paths.items()},
        "instances": rows_out,
    }
    (out_dir / "trusted_reasons_build.json").write_text(
        json.dumps(build_summary, indent=2))
    print(f"saved: {out_dir / 'trusted_reasons_build.json'}")


def evaluate():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--arm-results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(sys.argv[2:])

    build = json.load(open(Path(args.build_dir) / "trusted_reasons_build.json"))
    arm = json.load(open(args.arm_results))
    gt_vec = {r["sample_id"]: r["x_gt"] for r in build["instances"]}
    default = build["default_row"]

    ok = [r for r in arm["results"] if r.get("status") == "ok"]
    excl_rows = [r for r in ok if "abstained" in r]
    covered = [r for r in excl_rows if not r["abstained"]]
    leaks = [r for r in covered if r.get("excluded_leak")]

    def clause_ok(lit, xg):
        return bool(xg[abs(lit) - 1] == (1 if lit > 0 else 0))

    all_correct = sum(
        1 for r in covered
        if all(clause_ok(l, gt_vec[r["sample_id"]]) for l in r["excluded_literals"]))
    # independence prediction at the same reason composition
    val_acc = {build["feature_names"].index(n): d["accuracy"]
               for n, d in build["validation_reliability"].items()
               if d["accuracy"] is not None}
    ind = [float(np.prod([val_acc.get(abs(l) - 1, 0.5)
                          for l in r["excluded_literals"]]))
           for r in covered if r["excluded_literals"]]
    mean_size = (float(np.mean([r["excluded_size"] for r in covered]))
                 if covered else None)

    report = {
        "table": {
            "default_reason": {
                "coverage": default["coverage"],
                "p_all_clauses_correct": default["p_all_clauses_correct"],
                "independence_prediction": default["independence_prediction"],
                "mean_reason_size": default["mean_reason_size"]},
            "trusted_only": {
                "coverage": len(covered) / len(excl_rows) if excl_rows else None,
                "n": len(excl_rows),
                "p_all_clauses_correct_given_covered":
                    all_correct / len(covered) if covered else None,
                "independence_prediction_given_covered":
                    float(np.mean(ind)) if ind else None,
                "mean_reason_size": mean_size},
        },
        "e1_output_match": build["e1_output_match"],
        "abstained": len(excl_rows) - len(covered),
        "leaks": len(leaks),
        "audit_native_is_sufficient": sum(
            1 for r in ok if r.get("native_is_sufficient_per_pyxai")),
        "n_ok": len(ok),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["table"], indent=1))
    print(f"E1 output-match: {report['e1_output_match']:.3f} | "
          f"leaks: {leaks and len(leaks)} | native audit: "
          f"{report['audit_native_is_sufficient']}/{report['n_ok']}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    {"build": build, "evaluate": evaluate}[sys.argv[1]]()
