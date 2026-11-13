"""
altbank.py — alternative feature banks from the same DINOv3 caches.
=====================================================================
Builds NON-semantic feature banks in the exact bank_{train,test}.npz
format so the whole pipeline (teacher/distill/explain/analysis/conaxp)
runs unchanged, isolating the FEATURE VECTOR as the only difference:

    kind=cls       global DINOv3 cls tokens (384-d)
    kind=meanpatch feature-map mean over patches (384-d)

Per bank: PCA fitted on TRAIN (all classes), --dims components, min-max
rescaled to [0,1] using train statistics (test clipped).  "concepts"
are then pc0..pc{d-1} — no semantics, no annotations:
    scores = the [0,1] features   (continuous; box-LP still exact)
    bits   = score >= 0.5          (Boolean student input, same protocol)
    cal    = scores                (no calibration layer)
    gt     = zeros                 (premise audit not applicable)

Usage
-----
    python src/altbank.py --kind cls --dims 32 --cache-dir <dino cache>

Output: intermediate_results/bank_{kind}/ (bank_train.npz, bank_test.npz,
meta.json).  Env: conda (sklearn PCA).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"


def build_alt_bank(cache_dir, kind, dims=32, out_dir=None):
    assert kind in ("cls", "meanpatch")
    out = Path(out_dir) if out_dir else INTERMEDIATE_DIR / f"bank_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    dtr = np.load(Path(cache_dir) / "train_features.npz")
    dte = np.load(Path(cache_dir) / "test_features.npz")

    def feats(d):
        if kind == "cls":
            return d["cls_tokens"].astype(np.float64)
        fm = d["feature_maps"].astype(np.float64)
        return fm.mean(axis=(2, 3))

    Xtr, Xte = feats(dtr), feats(dte)
    pca = PCA(n_components=dims, random_state=42).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    lo, hi = Ptr.min(axis=0), Ptr.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Str = np.clip((Ptr - lo) / span, 0, 1)
    Ste = np.clip((Pte - lo) / span, 0, 1)
    evr = float(pca.explained_variance_ratio_.sum())

    for split, S, d in (("train", Str, dtr), ("test", Ste, dte)):
        np.savez(out / f"bank_{split}.npz",
                 sample_ids=d["sample_ids"].astype(np.int64),
                 labels=d["labels"].astype(np.int64),
                 bits=(S >= 0.5).astype(np.int8),
                 scores=S.astype(np.float32),
                 cal=S.astype(np.float32),
                 gt=np.zeros((len(S), dims), dtype=np.int8))
    (out / "meta.json").write_text(json.dumps({
        "feature_names": [f"pc{i}" for i in range(dims)],
        "feat_part": [""] * dims, "part_feat_idx": [], "vocab": [],
        "protocol": {
            "kind": kind, "dims": dims,
            "cache": str(Path(cache_dir)),
            "rescale": ("PCA on train (all classes), min-max to [0,1] "
                        "with train statistics; bits = score >= 0.5"),
            "explained_variance_ratio_sum": round(evr, 4),
            "n_train": int(len(Str)), "n_test": int(len(Ste)),
            "n_features": dims,
            "semantic": False,
        },
    }, indent=2))
    print(f"--> alt bank [{kind}] -> {out}: F={dims} "
          f"(PCA EVR {evr:.3f}) train={len(Str)} test={len(Ste)}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--kind", choices=["cls", "meanpatch"], required=True)
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build_alt_bank(args.cache_dir, args.kind, dims=args.dims, out_dir=args.out)


if __name__ == "__main__":
    main()
