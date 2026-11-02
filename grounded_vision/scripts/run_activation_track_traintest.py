"""
Script: run_activation_track_traintest.py
==========================================
Activation-grounding track under the OFFICIAL CUB train/test split —
replaces the original 100-image in-cache protocol.

Protocol:
- Load cached DINOv2 features for BOTH splits (cache_features.py --split all).
- Pool 4x4 tile x 384 channels with the CORRECTED within-tile axes (3,5).
- Fit the same tree family as the original track (depth-6 CART, Gini,
  min_samples_leaf=2, seed 42) on TRAIN only.
- Evaluate on TEST only: accuracy, verified sufficient-reason rate, mean
  reason size, grounding-selected reasons (part-score maximization over
  sufficient subsets), TreeSHAP top-|reason| overlap, part-grounded NLX
  examples, and train->test exemplar retrieval for certified tiles.

Outputs (under --run-dir):
- activation_track_traintest_results.json
- activation_traintest_selected_reasons.json
- activation_traintest_examples.png     (certified tiles on test images)
- activation_traintest_exemplars.png    (nearest TRAIN tile exemplars)
"""

from __future__ import annotations
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier
from grounded_vision.explanations.schema import ExplanationCondition, VerificationStatus


def pool(fmap: np.ndarray) -> np.ndarray:
    """[N,384,16,16] -> [N, 16*384]; mean over WITHIN-tile axes (3,5)."""
    return (
        fmap.reshape(len(fmap), 384, 4, 4, 4, 4)
        .mean((3, 5))
        .transpose(0, 2, 3, 1)
        .reshape(len(fmap), -1)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="directory holding {train,test}_features.npz")
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--n-reasons", type=int, default=500,
                    help="test samples for reason/grounding/SHAP analysis")
    ap.add_argument("--max-classes", type=int, default=0,
                    help="restrict to the first K species (0 = all 200); "
                         "K=4 reproduces the historical subset protocol "
                         "under a proper train/test split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run, cub = Path(args.run_dir), Path(args.cub_dir)
    dtr = np.load(run / "train_features.npz")
    dte = np.load(run / "test_features.npz")
    fm_tr, fm_te = dtr["feature_maps"], dte["feature_maps"]  # materialize once
    ytr, yte = dtr["labels"], dte["labels"]
    ids_te = dte["sample_ids"]
    ids_tr = dtr["sample_ids"]
    if args.max_classes:
        mtr = ytr < args.max_classes
        mte = yte < args.max_classes
        fm_tr, ytr, ids_tr = fm_tr[mtr], ytr[mtr], ids_tr[mtr]
        fm_te, yte, ids_te = fm_te[mte], yte[mte], ids_te[mte]
        print(f"--> restricted to first {args.max_classes} species: "
              f"train {int(mtr.sum())} / test {int(mte.sum())} images",
              flush=True)
    xtr, xte = pool(fm_tr), pool(fm_te)
    row_of_sid = {int(s): i for i, s in enumerate(ids_te)}
    print(f"--> train {len(ytr)} / test {len(yte)} images "
          f"({xtr.shape[1]} pooled features)", flush=True)

    names = [f"tile_r{r}_c{c}:dino_ch{k}"
             for r in range(4) for c in range(4) for k in range(384)]
    tree = CertifiableTreeHead(max_depth=6, min_samples_leaf=2,
                               feature_names=names)
    tree.fit(xtr, ytr)
    pred_tr, pred_te = tree.predict(xtr), tree.predict(xte)
    acc_tr = float(np.mean(pred_tr == ytr))
    acc_te = float(np.mean(pred_te == yte))
    n_classes_te = len(np.unique(yte))
    print(f"--> train acc {acc_tr:.4f} | test acc {acc_te:.4f} "
          f"({n_classes_te} classes present in test)", flush=True)

    backend = PyXAIBackend(tree)
    verifier = TreeSufficiencyVerifier(tree.clf, names)

    # ---- CUB annotations + image sizes (header-only PIL opens) --------------
    paths = {int(a): b.strip() for a, b in (
        l.split(maxsplit=1) for l in (cub / "images.txt").read_text().splitlines())}
    pnames = {int(z.split()[0]): z.split(maxsplit=1)[1]
              for z in (cub / "parts/parts.txt").read_text().splitlines()}
    parts = {}
    for z in (cub / "parts/part_locs.txt").read_text().splitlines():
        s, p, a, b, vis = z.split()
        if int(vis):
            parts.setdefault(int(s), []).append(
                (pnames[int(p)], float(a), float(b)))
    size_cache = {}

    def size_of(sid):
        if sid not in size_cache:
            with Image.open(cub / "images" / paths[sid]) as im:
                size_cache[sid] = im.size  # (W, H)
        return size_cache[sid]

    def parts_in_tile(sid, feature_idx):
        """Visible CUB part names whose (pixel) point lies inside the tile."""
        w, h = size_of(sid)
        tile = feature_idx // 384
        r, col = divmod(tile, 4)
        return sorted({n for n, px, py in parts.get(sid, [])
                       if col / 4 <= px / w <= (col + 1) / 4
                       and r / 4 <= py / h <= (r + 1) / 4})

    # ---- test reasons + grounding selection ---------------------------------
    rng = np.random.RandomState(args.seed)
    rows = np.sort(rng.permutation(len(xte))[: min(args.n_reasons, len(xte))])
    sizes, verified, selected_rows, sel_pos = [], [], [], {}
    for i in rows:
        x, pred = xte[i], int(pred_te[i])
        exp = backend.explain_sufficient(x, pred, sample_id=int(i))
        ok = exp.certificate.status == VerificationStatus.VERIFIED
        verified.append(ok)
        if not ok:
            continue
        sizes.append(len(exp.conditions))
        sid = int(ids_te[i])
        conds = {c.feature_idx: c for c in exp.conditions}
        for q in tree.get_decision_path(x):
            conds.setdefault(q.feature_idx, ExplanationCondition(
                q.feature_idx, q.feature_name,
                "<=" if q.is_left else ">", q.threshold, q.value))
        path = list(conds.values())
        valid = []
        for n in range(1, len(path) + 1):
            for sub in itertools.combinations(path, n):
                if (verifier.verify_sufficiency(list(sub), pred).status
                        == VerificationStatus.VERIFIED):
                    valid.append(list(sub))
        bestn = min(map(len, valid))
        cand = [a for a in valid if len(a) <= bestn + 1]
        chosen = max(cand, key=lambda a: len({p for c in a
                                              for p in parts_in_tile(sid, c.feature_idx)}))
        selected_rows.append({
            "sample_id": sid, "row": int(i), "prediction": pred,
            "shortest_size": bestn, "selected_size": len(chosen),
            "part_score": len({p for c in chosen
                               for p in parts_in_tile(sid, c.feature_idx)}),
            "conditions": [c.to_dict() for c in chosen],
        })
        sel_pos[int(i)] = selected_rows[-1]
    verified_rate = float(np.mean(verified))
    mean_size = float(np.mean(sizes)) if sizes else None
    mean_selected = float(np.mean([r["selected_size"] for r in selected_rows]))
    mean_part = float(np.mean([r["part_score"] for r in selected_rows]))
    print(f"--> reasons (n={len(rows)}): verified {verified_rate:.4f}, "
          f"mean size {mean_size:.2f}; selected {mean_selected:.2f}, "
          f"mean part score {mean_part:.2f}", flush=True)

    # ---- TreeSHAP overlap at matched length (chunked: 200-class tree) -------
    import shap
    explainer = shap.TreeExplainer(tree.clf)
    overlaps = {}
    chunk = 25
    for start in range(0, len(rows), chunk):
        idx = rows[start : start + chunk]
        sv = explainer.shap_values(xte[idx])
        p_idx = pred_te[idx]
        if isinstance(sv, list):
            vals = np.stack(sv, axis=1)[np.arange(len(idx)), p_idx]
        else:
            vals = sv[np.arange(len(idx)), :, p_idx] if sv.ndim == 3 else sv
        for j, i in enumerate(idx):
            r = sel_pos.get(int(i))
            if r is None:
                continue
            sel = {c["feature_idx"] for c in r["conditions"]}
            top = set(np.argsort(-np.abs(vals[j]))[: len(sel)].tolist())
            overlaps[r["sample_id"]] = len(sel & top) / max(1, len(sel))
        del sv, vals
    shap_overlap = float(np.mean(list(overlaps.values())))
    print(f"--> TreeSHAP top-|reason| overlap: {shap_overlap:.4f}", flush=True)

    # ---- part-grounded NLX examples ------------------------------------------
    examples = []
    for r in selected_rows:
        if len(examples) >= 4:
            break
        clauses = []
        for c in r["conditions"]:
            ch = int(c["feature_idx"]) % 384
            tile = c["feature_idx"] // 384
            rr, cc = divmod(tile, 4)
            pn = parts_in_tile(r["sample_id"], c["feature_idx"])
            where = (", ".join(pn) + " area") if pn else \
                f"tile (row {rr + 1}, column {cc + 1})"
            op = "<=" if c["operator"] == "<=" else ">"
            clauses.append(f"activation channel {ch} in the {where} "
                           f"is {op} {c['threshold']:.3f}")
        examples.append({
            "sample_id": r["sample_id"], "prediction": r["prediction"],
            "text": f"The activation tree predicts class {r['prediction']} "
                    f"because " + "; and ".join(clauses) + ".",
        })

    # ---- evidence PNG on test images ----------------------------------------
    by_class = {}
    for i in rows:
        if pred_te[i] == yte[i]:
            by_class.setdefault(int(pred_te[i]), []).append(i)
    chosen_img = [v[0] for _, v in sorted(by_class.items())][:8]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, i in zip(axs.flat, chosen_img):
        sid = int(ids_te[i])
        im = Image.open(cub / "images" / paths[sid]).convert("RGB")
        w, h = im.size
        ax.imshow(im)
        ax.set_title(f"sample {sid} -> class {int(pred_te[i])} (ok)",
                     fontsize=9)
        exp = backend.explain_sufficient(xte[i], int(pred_te[i]), sample_id=i)
        for g in exp.conditions:
            r, col = divmod(g.feature_idx // 384, 4)
            ax.add_patch(plt.Rectangle((col * w / 4, r * h / 4),
                                       w / 4, h / 4, fill=False,
                                       ec="#ff7f0e", lw=2))
    for ax in axs.flat[len(chosen_img):]:
        ax.axis("off")
    fig.suptitle("Activation-tree sufficient reasons on TEST images "
                 "(orange = certified tiles)", fontweight="bold")
    fig.tight_layout()
    tag = f"_c{args.max_classes}" if args.max_classes else ""
    fig.savefig(run / f"activation_traintest_examples{tag}.png", dpi=150)
    plt.close(fig)

    # ---- exemplar retrieval: test tile -> nearest TRAIN tile -----------------
    emb_tr = xtr.reshape(len(xtr), 16, 384)
    emb_tr = emb_tr / np.maximum(np.linalg.norm(emb_tr, axis=-1,
                                                keepdims=True), 1e-8)
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    k = 0
    for r in selected_rows[:4]:
        sid, c0 = r["sample_id"], r["conditions"][0]
        tile = c0["feature_idx"] // 384
        rr, cc = divmod(tile, 4)
        q = xte[r["row"]].reshape(16, 384)[tile]
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = emb_tr[:, tile, :] @ q
        best = int(np.argmax(sims))
        bsid = int(ids_tr[best])
        ax_q, ax_e = axs.flat[2 * k], axs.flat[2 * k + 1]
        im_q = Image.open(cub / "images" / paths[sid]).convert("RGB")
        w, h = im_q.size
        ax_q.imshow(im_q.crop((cc * w / 4, rr * h / 4,
                               (cc + 1) * w / 4, (rr + 1) * h / 4)))
        ax_q.set_title(f"TEST {sid} tile r{rr}c{cc}", fontsize=8)
        im_e = Image.open(cub / "images" / paths[bsid]).convert("RGB")
        w, h = im_e.size
        ax_e.imshow(im_e.crop((cc * w / 4, rr * h / 4,
                               (cc + 1) * w / 4, (rr + 1) * h / 4)))
        ax_e.set_title(f"TRAIN nn {bsid} ({sims[best]:.2f})", fontsize=8)
        k += 1
    for ax in axs.flat[2 * k:]:
        ax.axis("off")
    fig.suptitle("Certified TEST tiles and nearest TRAIN-tile exemplars",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(run / f"activation_traintest_exemplars{tag}.png", dpi=150)
    plt.close(fig)

    report = {
        "protocol": {
            "split": "official CUB train_test_split.txt",
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "classes_in_test": int(n_classes_te),
            "tree": "CART depth 6, gini, min_samples_leaf 2, seed 42",
            "pooling": "4x4 tiles x 384 channels, mean over within-tile axes (3,5)",
            "reason_samples": int(len(rows)),
            "seed": args.seed,
        },
        "train_accuracy": acc_tr,
        "test_accuracy": acc_te,
        "verified_rate_on_test": verified_rate,
        "mean_reason_size_on_test": mean_size,
        "grounding_selected": {
            "mean_shortest": float(np.mean([r["shortest_size"]
                                            for r in selected_rows])),
            "mean_selected": mean_selected,
            "mean_part_score": mean_part,
        },
        "shap_overlap_on_test": shap_overlap,
        "nlx_examples": examples,
    }
    out = run / f"activation_track_traintest_results{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    (run / f"activation_traintest_selected_reasons{tag}.json").write_text(
        json.dumps(selected_rows, indent=2))
    print(f"saved: {out}", flush=True)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "nlx_examples"}, indent=1))


if __name__ == "__main__":
    main()
