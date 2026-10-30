"""
Script: run_matryoshka_sae.py
=============================
Certified explanations over sparse SAE latents of the DINO tile activations.

Pipeline:
  1. Pool cached DINO feature maps into 4x4 tiles (as the activation track),
     L2-normalize each tile vector.
  2. Train a small TopK Matryoshka SAE (nested prefixes LEVELS = 32/128/512):
     one encoder, TopK codes, per-prefix reconstruction losses — coarse
     prefixes must reconstruct on their own.
  3. For each level (and a dense-channel baseline under the identical
     protocol): fit a CART tree on TRAIN codes, report TEST accuracy, and
     extract certified sufficient reasons on a fixed 200-instance test sample
     with the native verifier (necessity by construction).
  4. Grounding: reason tiles -> visible CUB part points (part-score).
  5. Semantic validation: latent-vs-CUB-attribute correlations (image-level),
     best attribute per latent, fraction of reason latents with |r| >= 0.3.
  6. Abstraction invariance: for the finest tree, is the coarse-prefix subset
     of a fine sufficient reason still sufficient? (Verified per instance.)
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sweep_tree_depth import tree_nodes  # noqa: E402
from grounded_vision.reasoning.tree_head import CertifiableTreeHead  # noqa: E402
from grounded_vision.explanations.pyxai_backend import PyXAIBackend  # noqa: E402
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier  # noqa: E402
from grounded_vision.explanations.schema import ExplanationCondition, VerificationStatus  # noqa: E402

CUB = "/Users/cril/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
RUN = "/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_full"
OUT = "/Users/cril/tanmoy/research/data/grounded_vision_runs/matryoshka_sae"
LEVELS = [32, 128, 512]
TOPK = 32
EPOCHS = 30
BATCH = 2048


def tile_pool(fmaps: np.ndarray) -> np.ndarray:
    """[N, 384, 16, 16] -> [N, 16, 384] (4x4 tiles, block means), L2-normalized."""
    n = fmaps.shape[0]
    t = fmaps.reshape(n, 384, 4, 4, 4, 4).mean(axis=(4, 5)).transpose(0, 2, 3, 1)
    t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-8)
    return t.reshape(n, 16, 384)


def train_msae(X: np.ndarray, m: int = 512, seed: int = 42) -> dict:
    """TopK Matryoshka SAE on unit-norm vectors. Returns encoder/decoder."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).to(dev)
    n, d = Xt.shape
    enc = torch.nn.Linear(d, m, bias=True).to(dev)
    dec = torch.nn.Linear(m, d, bias=False).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    lvl_w = {32: 4.0, 128: 2.0, 512: 1.0}
    rng = np.random.RandomState(seed)
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, BATCH):
            idx = torch.from_numpy(perm[s:s + BATCH]).to(dev)
            x = Xt[idx]
            z_pre = enc(x)
            # global TopK over the m latents
            kth = torch.kthvalue(z_pre, m - TOPK + 1, dim=1).values.unsqueeze(1)
            z = torch.where(z_pre > kth, z_pre, torch.zeros_like(z_pre))
            loss = 0.0
            for lvl in LEVELS:
                rec = z[:, :lvl] @ dec.weight[:, :lvl].T
                loss = loss + lvl_w[lvl] * ((rec - x) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # standard TopK-SAE constraint: unit-norm decoder columns
            with torch.no_grad():
                dec.weight.data = F.normalize(dec.weight.data, dim=0)
            tot += float(loss.item()) * len(idx)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  SAE epoch {epoch+1}/{EPOCHS} - loss {tot/n:.6f}")
    return {"enc_w": enc.weight.detach().cpu().numpy(), "enc_b": enc.bias.detach().cpu().numpy(),
            "dec_w": dec.weight.detach().cpu().numpy()}


def encode(msae, X: np.ndarray) -> np.ndarray:
    z = X @ msae["enc_w"].T + msae["enc_b"]
    kth = np.partition(z, z.shape[1] - TOPK, axis=1)[:, z.shape[1] - TOPK][:, None]
    z = np.where(z > kth, z, 0.0).astype(np.float32)
    return z


def fit_and_explain(Xtr, ytr, Xte, yte, feature_names, rows, max_depth=6):
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=2, random_state=42)
    dt.fit(Xtr, ytr)
    head = CertifiableTreeHead(max_depth=max_depth, min_samples_leaf=2,
                               feature_names=feature_names)
    head.clf, head.is_fitted = dt, True
    backend = PyXAIBackend(head)
    preds = dt.predict(Xte)
    reasons, path_lens, conds = [], [], []
    for i in rows:
        x = Xte[i]
        pred = int(preds[i])
        exp = backend.explain_sufficient(x, pred, sample_id=int(i))
        reasons.append([(c.feature_idx, c.operator, c.threshold) for c in exp.conditions])
        path_lens.append(len(head.get_decision_path(x)))
        con = backend.explain_contrastive(x, pred, sample_id=int(i))
        _ = con
    return {
        "dt": dt, "head": head,
        "train_acc": float(np.mean(dt.predict(Xtr) == ytr)),
        "test_acc": float(np.mean(preds == yte)),
        "reasons": reasons, "path_lengths": path_lens,
        "reason_mean": float(np.mean([len(r) for r in reasons])),
        "reason_max": int(np.max([len(r) for r in reasons])),
        "frac_reason_lt_path": float(np.mean([len(r) < p for r, p in zip(reasons, path_lens)])),
    }


def load_part_points():
    names = {int(l.split()[0]): l.split(maxsplit=1)[1].strip()
             for l in open(f"{CUB}/parts/parts.txt").read().splitlines() if l.strip()}
    sizes = {}
    imgs = {int(l.split()[0]): l.split(maxsplit=1)[1].strip()
            for l in open(f"{CUB}/images.txt").read().splitlines() if l.strip()}
    from PIL import Image
    pts = {}
    for line in open(f"{CUB}/parts/part_locs.txt").read().splitlines():
        f = line.split()
        if len(f) != 5 or not int(f[4]):
            continue
        sid, pid, x, y = int(f[0]), int(f[1]), float(f[2]), float(f[3])
        if sid not in sizes:
            with Image.open(f"{CUB}/images/{imgs[sid]}") as im:
                sizes[sid] = im.size
        w, h = sizes[sid]
        pts.setdefault(sid, []).append((names[pid], x / w, y / h))
    return pts


def part_score(reasons, rows, test_ids, pts, stride):
    """Mean number of distinct visible parts covered by reason tiles."""
    scores = []
    for r, i in zip(reasons, rows):
        found = set()
        for (fidx, op, thr) in r:
            tile = fidx // stride
            row, col = divmod(tile, 4)
            for n, px, py in pts.get(int(test_ids[i]), []):
                if col / 4 <= px <= (col + 1) / 4 and row / 4 <= py <= (row + 1) / 4:
                    found.add(n)
        scores.append(len(found))
    return float(np.mean(scores))


def main():
    os.makedirs(OUT, exist_ok=True)
    dtr = np.load(f"{RUN}/train_features.npz")
    dte = np.load(f"{RUN}/test_features.npz")
    ytr, yte = dtr["labels"], dte["labels"]
    ids_tr, ids_te = dtr["sample_ids"], dte["sample_ids"]

    Ttr, Tte = tile_pool(dtr["feature_maps"]), tile_pool(dte["feature_maps"])
    print(f"tiles: train {Ttr.shape}, test {Tte.shape}")

    print("--> Training Matryoshka SAE...")
    msae = train_msae(Ttr.reshape(-1, 384))
    Ztr = encode(msae, Ttr.reshape(-1, 384)).reshape(len(ytr), 16, -1)
    Zte = encode(msae, Tte.reshape(-1, 384)).reshape(len(yte), 16, -1)
    sparsity = float((Ztr > 0).mean())
    # reconstruction quality per level
    Xflat = Ttr.reshape(-1, 384)
    Zflat = Ztr.reshape(-1, Ztr.shape[-1])
    r2 = {}
    for lvl in LEVELS:
        rec = Zflat[:, :lvl] @ msae["dec_w"][:, :lvl].T
        r2[lvl] = float(1 - ((rec - Xflat) ** 2).sum() / (Xflat ** 2).sum())
    print(f"  active-rate {sparsity:.4f} | R^2 per level: " +
          ", ".join(f"L{k}={v:.3f}" for k, v in r2.items()))

    rng = np.random.RandomState(42)
    rows = rng.choice(len(yte), size=200, replace=False)
    pts = load_part_points()

    results = {"levels": LEVELS, "topk": TOPK, "sparsity": sparsity, "r2": r2, "configs": {}}

    # ---- Dense-channel baseline (identical protocol) -------------------------
    print("--> Dense-channel baseline tree...")
    Xtr_c = Ttr.reshape(len(ytr), -1)
    Xte_c = Tte.reshape(len(yte), -1)
    base = fit_and_explain(Xtr_c, ytr, Xte_c, yte,
                           [f"t{t}:ch{c}" for t in range(16) for c in range(384)], rows)
    results["configs"]["channels"] = {
        "train_acc": base["train_acc"], "test_acc": base["test_acc"],
        "reason_mean": base["reason_mean"], "reason_max": base["reason_max"],
        "frac_reason_lt_path": base["frac_reason_lt_path"],
        "part_score": part_score(base["reasons"], rows, ids_te, pts, stride=384),
    }
    print(f"  channels: test {base['test_acc']*100:.2f}% | reason {base['reason_mean']:.2f} "
          f"| parts {results['configs']['channels']['part_score']:.2f}")

    # ---- SAE levels ----------------------------------------------------------
    sae_reasons_fine = None
    for lvl in LEVELS:
        names = [f"L{lvl}:t{t}:z{z}" for t in range(16) for z in range(lvl)]
        Xtr_l = Ztr[:, :, :lvl].reshape(len(ytr), -1)
        Xte_l = Zte[:, :, :lvl].reshape(len(yte), -1)
        res = fit_and_explain(Xtr_l, ytr, Xte_l, yte, names, rows)
        entry = {
            "train_acc": res["train_acc"], "test_acc": res["test_acc"],
            "reason_mean": res["reason_mean"], "reason_max": res["reason_max"],
            "frac_reason_lt_path": res["frac_reason_lt_path"],
        }
        if lvl == LEVELS[-1]:
            sae_reasons_fine = res
            entry["part_score"] = part_score(res["reasons"], rows, ids_te, pts, stride=lvl)
        results["configs"][f"sae_{lvl}"] = entry
        print(f"  SAE L{lvl}: test {res['test_acc']*100:.2f}% | reason {res['reason_mean']:.2f} "
              f"(max {res['reason_max']}) | compr {res['frac_reason_lt_path']*100:.0f}%"
              + (f" | parts {entry['part_score']:.2f}" if "part_score" in entry else ""))

    # ---- Semantic validation: latent <-> CUB attribute ------------------------
    from run_312_ceiling import load_full_attributes
    _, gt_by_id = load_full_attributes(CUB)
    L = np.stack([gt_by_id[int(s)] for s in ids_tr])          # [N, 312]
    Zimg = Ztr.mean(axis=1)                                    # [N, m] image-mean activations
    Zc = Zimg - Zimg.mean(0)
    Lc = L - L.mean(0)
    corr = (Zc.T @ Lc) / np.maximum(np.outer(Zimg.std(0), L.std(0)) * len(L), 1e-8)
    best_attr = np.abs(corr).argmax(axis=1)
    best_r = np.abs(corr)[np.arange(len(best_attr)), best_attr]
    validated = set(np.where(best_r >= 0.3)[0].tolist())

    fine = LEVELS[-1]
    used_latents = set()
    for r in sae_reasons_fine["reasons"]:
        for (fidx, op, thr) in r:
            used_latents.add(fidx % fine)
    entry = results["configs"][f"sae_{fine}"]
    entry["n_distinct_reason_latents"] = len(used_latents)
    entry["validated_reason_latents_r>=0.3"] = len(used_latents & validated)
    results["latent_validation"] = {
        "n_latents": fine, "n_validated_r>=0.3": len(validated),
        "examples": [
            {"latent": int(j), "best_r": round(float(best_r[j]), 3),
             "best_attr_idx": int(best_attr[j])}
            for j in list(np.argsort(-best_r)[:10])
        ],
    }

    # ---- Abstraction invariance: coarse-prefix subset of fine reason ----------
    print("--> Abstraction-invariance check (fine tree, coarse-prefix subsets)...")
    coarse = LEVELS[0]
    verifier = TreeSufficiencyVerifier(sae_reasons_fine["head"].clf,
                                       sae_reasons_fine["head"].feature_names)
    inv_ok, inv_total, inv_sizes = 0, 0, []
    for r, i in zip(sae_reasons_fine["reasons"], rows):
        pred = int(sae_reasons_fine["dt"].predict(
            Zte[:, :, :fine].reshape(len(yte), -1)[i].reshape(1, -1))[0])
        subset = [(fidx, op, thr) for (fidx, op, thr) in r if (fidx % fine) < coarse]
        if not subset:
            continue
        inv_total += 1
        conds = [ExplanationCondition(f, hname, op, thr, observed_value=0.0)
                 for (f, hname, op, thr) in
                 [(fidx, sae_reasons_fine["head"].feature_names[fidx], op, thr)
                  for (fidx, op, thr) in subset]]
        cert = verifier.verify_sufficiency(conds, pred)
        if cert.status == VerificationStatus.VERIFIED:
            inv_ok += 1
            inv_sizes.append(len(subset))
    results["abstraction_invariance"] = {
        "coarse_level": coarse,
        "instances_with_coarse_subset": inv_total,
        "coarse_subset_sufficient_rate": inv_ok / max(1, inv_total),
        "mean_coarse_subset_size_when_sufficient":
            float(np.mean(inv_sizes)) if inv_sizes else None,
    }
    print(f"  coarse-prefix subset sufficient: {inv_ok}/{inv_total}")

    # ---- Export numeric PyXAI payload for the fine tree ------------------------
    dt = sae_reasons_fine["dt"]
    tree = dt.tree_
    nodes = []
    for nid in range(tree.node_count):
        if tree.feature[nid] == -2:
            nodes.append({"f": -1, "l": -1, "r": -1, "c": int(np.argmax(tree.value[nid][0])), "t": 0.0})
        else:
            nodes.append({"f": int(tree.feature[nid]), "l": int(tree.children_left[nid]),
                          "r": int(tree.children_right[nid]), "c": -1,
                          "t": float(tree.threshold[nid])})
    Xfine_te = Zte[:, :, :fine].reshape(len(yte), -1)
    instances = []
    for r, i in zip(sae_reasons_fine["reasons"], rows):
        instances.append({
            "sample_id": int(i), "x": [float(v) for v in Xfine_te[i]],
            "prediction": int(dt.predict(Xfine_te[i].reshape(1, -1))[0]),
            "label": int(yte[i]),
            "native_reason_literals": [c.condition_str for c in
                                       PyXAIBackend(sae_reasons_fine["head"]).explain_sufficient(
                                           Xfine_te[i], int(dt.predict(Xfine_te[i].reshape(1, -1))[0]),
                                           sample_id=int(i)).conditions],
            "native_reason_size": len(r), "path_length": sae_reasons_fine["path_lengths"][len(instances)],
        })
    with open(f"{OUT}/msae_fine_pyxai.json", "w") as f:
        json.dump({"n_features": int(Xfine_te.shape[1]), "numeric": True,
                   "feature_names": [f"t{t}:z{z}" for t in range(16) for z in range(fine)],
                   "nodes": nodes, "instances": instances}, f)

    with open(f"{OUT}/msae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2)[:2000])
    print(f"saved: {OUT}/msae_results.json")


if __name__ == "__main__":
    main()
