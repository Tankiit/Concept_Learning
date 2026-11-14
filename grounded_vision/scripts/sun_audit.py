"""SUN Attribute DB premise-support audit (same recipe as CelebA/BDD-OIA).

Images: official SUNAttributeDB_Images archive; labels: labels_cv (14340x102
continuous, thresholded at 0.5); NO official split exists -> category-stratified
70/15/15 carve, seed 42 (stated in all write-ups). Readers: logistic on DINOv2
vits14 features over the full train carve (all 611 categories); per-K subset
task = classify top-K categories from 102 concept bits; teacher = multinomial
logistic on bits; student = CART depth 4; native TreeOps minimal reasons;
premise support vs thresholded GT + A/B/C abstention arms.

Env: conda torch-multimodal (torch+sklearn+scipy). Artifacts: sun_audit_K{4,10}.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

RUN = Path(__file__).parent
SUN = RUN.parent / "datasets" / "sun"
sys.path.insert(0, str(RUN.parent.parent / "bankexp"))
from utils.tree_ops import TreeOps  # noqa: E402

SEED = 42
THR = 0.5
TOP_CATS = ["underwater", "waterfall", "stadium", "theater", "arena",
            "elevator", "temple", "train_station", "airport", "apse"]
K_LIST = [4, 10]


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = (z / (1 + z * z / n)) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0., c - h), min(1., c + h))


def features():
    """DINOv2 ViT-S/14 CLS features for all indexed images, cached."""
    cache = RUN / "feats_sun.npy"
    if cache.exists():
        return np.load(cache)
    import torch
    import torchvision.transforms as T
    from PIL import Image
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    tfm = T.Compose([T.Resize((224, 224), antialias=True), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    imgs = [i[0][0] for i in loadmat(SUN / "SUNAttributeDB/images.mat")["images"]]
    feats = np.zeros((len(imgs), 384), dtype=np.float32)
    B = 128
    with torch.no_grad():
        for i in range(0, len(imgs), B):
            batch = [tfm(Image.open(SUN / "images" / p).convert("RGB"))
                     for p in imgs[i:i + B]]
            feats[i:i + len(batch)] = model(torch.stack(batch).to(device)).cpu().numpy()
            if (i // B) % 25 == 0:
                print(f"  feats {i + len(batch)}/{len(imgs)}", flush=True)
    np.save(cache, feats)
    return feats


def main():
    rng = np.random.RandomState(SEED)
    X = features()
    labels = loadmat(SUN / "SUNAttributeDB/attributeLabels_continuous.mat")["labels_cv"]
    imgs = [i[0][0] for i in loadmat(SUN / "SUNAttributeDB/images.mat")["images"]]
    G = labels >= THR                       # ground-truth bits
    cats = np.array([p.split("/")[1] for p in imgs])

    # --- category-stratified carve (no official split exists) ---
    split = np.zeros(len(imgs), dtype=int)  # 0 train, 1 val, 2 test
    for c in np.unique(cats):
        idx = np.where(cats == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_te, n_va = int(round(0.15 * n)), int(round(0.15 * n))
        split[idx[:n_te]] = 2
        split[idx[n_te:n_te + n_va]] = 1
    tr, va, te = split == 0, split == 1, split == 2
    print(f"carve: train {tr.sum()} / val {va.sum()} / test {te.sum()} (seed {SEED})")

    # --- readers over the full train carve ---
    readers, reader_va = {}, {}
    F = G.shape[1]
    for j in range(F):
        y = G[tr, j].astype(int)
        if y.min() == y.max():
            readers[j] = (None, float(y[0])); continue
        m = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        m.fit(X[tr], y)
        readers[j] = (m, None)
        reader_va[j] = float(m.score(X[va], G[va, j].astype(int)))
    accs = np.array(list(reader_va.values()))
    print(f"readers: {len(reader_va)}/{F} usable, val acc mean {accs.mean():.3f} "
          f"[{accs.min():.3f}, {accs.max():.3f}]")

    def probs(mask):
        P = np.full((mask.sum(), F), np.nan)
        for j, (m, const) in readers.items():
            if m is not None:
                P[:, j] = m.predict_proba(X[mask])[:, 1]
        return P

    Pva, Pte = probs(va), probs(te)
    Bva, Bte = Pva >= 0.5, Pte >= 0.5

    out_all = {}
    for K in K_LIST:
        sel = np.isin(cats, TOP_CATS[:K])
        cls = {c: i for i, c in enumerate(TOP_CATS[:K])}
        y_all = np.array([cls.get(c, -1) for c in cats])
        trk, vak, tek = tr & sel, va & sel, te & sel
        ytr, yva, yte = y_all[trk], y_all[vak], y_all[tek]

        teacher = LogisticRegression(max_iter=2000, random_state=SEED)
        # teacher over reader BITS of subset-train
        Btr = (probs(trk) >= 0.5)
        teacher.fit(Btr.astype(float), ytr)
        Ptk = probs(tek)                     # subset-test reader probs
        Btk = Ptk >= 0.5                     # subset-test bits
        Gt = G[tek]                          # subset-test ground truth
        tp = teacher.predict(Btk.astype(float))

        tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        tree.fit(Btr.astype(float), ytr)
        tj = RUN / f"tree_sun_K{K}.json"
        t = tree.tree_

        def rec(node):
            if t.children_left[node] == -1:
                return {"c": int(np.argmax(t.value[node]))}
            return {"f": int(t.feature[node]), "t": 0.5,
                    "l": rec(t.children_left[node]), "r": rec(t.children_right[node])}
        tj.write_text(json.dumps(dict(classes=list(range(K)), n_features=F,
                                      feature_names=[f"c{j}" for j in range(F)],
                                      root=rec(0))))
        ops = TreeOps(json.loads(tj.read_text()))
        sp = np.array([ops.predict_one(x) for x in Btk.astype(float)])

        te_idx = np.where(te)[0]               # global indices of test rows
        sub_local = np.where(tek[te_idx])[0]   # their positions within the test split
        per_instance, n_lit = [], []
        for k in range(Btk.shape[0]):
            reason = ops.minimal_reason(Btk[k].astype(float), int(sp[k]))
            lits = [(j, bool(v)) for j, v in reason]
            sup = [(bool(Gt[k, j]) == b) for j, b in lits]
            n_lit.append(len(lits))
            per_instance.append(dict(idx=int(sub_local[k]), pred=int(sp[k]),
                                     gt=int(yte[k]),
                                     reason=[[int(j), int(b)] for j, b in lits],
                                     all_true=bool(all(sup))))
        at = np.array([p["all_true"] for p in per_instance])
        n = len(at)
        p_all = float(at.mean()); lo, hi = wilson(p_all, n)

        lit_stats = {}
        for p in per_instance:
            k = p["idx"]
            for j, b in [(l[0], bool(l[1])) for l in p["reason"]]:
                t_ok = bool(G[te][k, j]) == b
                d = lit_stats.setdefault((int(j), b), [0, 0])
                d[0] += 1; d[1] += int(t_ok)
        marg = sorted(((k, v[1] / v[0], v[0]) for k, v in lit_stats.items()),
                      key=lambda x: -x[2] * (1 - x[1]))

        # abstention arms
        R = np.zeros((F, 2))
        for j in range(F):
            for b in (0, 1):
                m = Bva[:, j] == b
                R[j, b] = ((Bva[m, j] == G[va][m, j]).mean() if m.sum() else 1.0)
        iso = {}
        for j in range(F):
            ir = IsotonicRegression(out_of_bounds="clip")
            ok = np.isfinite(Pva[:, j])
            ir.fit(Pva[ok, j], G[va][ok, j].astype(int))
            iso[j] = ir

        def score(r, kind):
            vals = []
            for j, b in [(l[0], bool(l[1])) for l in r["reason"]]:
                if kind == "A":
                    vals.append(R[j, int(b)])
                else:
                    pj = float(np.clip(Pte[r["idx"], j], 1e-6, 1 - 1e-6))
                    if kind == "B":
                        cal = float(iso[j].predict([pj])[0])
                        vals.append(cal if b else 1 - cal)
                    else:
                        vals.append(pj if b else 1 - pj)
            return min(vals)

        curve = []
        for kind in ("A", "B", "C"):
            sc = np.array([score(r, kind) for r in per_instance])
            for cov in [1.0, 0.75, 0.5, 0.254, 0.136, 0.05]:
                n_cov = max(1, int(round(cov * n)))
                thr_ = np.partition(sc, -n_cov)[-n_cov]
                keep = sc >= thr_
                p_cov = float(at[keep].mean())
                lo_c, hi_c = wilson(p_cov, int(keep.sum()))
                rnd = float(np.mean([at[rng.choice(n, int(keep.sum()), replace=False)].mean()
                                     for _ in range(100)]))
                curve.append(dict(score=kind, coverage=float(keep.mean()),
                                  p_all_given_cov=p_cov, lo=lo_c, hi=hi_c,
                                  n=int(keep.sum()), random=rnd))

        res = dict(
            K=K, categories=TOP_CATS[:K],
            n=dict(train=int(trk.sum()), val=int(vak.sum()), test=n),
            reader_valid_acc_mean=float(accs.mean()) if len(accs) else None,
            teacher_acc=float((tp == yte).mean()),
            tree_acc=float((sp == yte).mean()),
            agreement=float((tp == sp).mean()),
            premise_support=dict(p=p_all, lo=lo, hi=hi, n=n),
            reason_len=dict(mean=float(np.mean(n_lit)), max=int(np.max(n_lit))),
            literal_marginals={f"c{j}={int(b)}": dict(n=c, support=float(s))
                               for (j, b), s, c in marg},
            abstention_curve=curve, per_instance=per_instance,
        )
        out_all[f"K{K}"] = res
        print(f"K={K}: n_test {n} | teacher {res['teacher_acc']:.3f} tree "
              f"{res['tree_acc']:.3f} agree {res['agreement']:.3f} | "
              f"P(all) {p_all:.3f} [{lo:.3f},{hi:.3f}] | len {np.mean(n_lit):.2f}")
        for (j, b), s, c in marg[:6]:
            print(f"   slot c{j}={'1' if b else '0'}: n={c} support={s:.3f}")
        for c_ in curve:
            if c_["score"] in ("B", "C"):
                print(f"   [{c_['score']}] cov={c_['coverage']:.3f}: "
                      f"P={c_['p_all_given_cov']:.3f} rnd={c_['random']:.3f}")

    out = dict(benchmark="SUN Attribute DB", seed=SEED, split="carved stratified 70/15/15",
               threshold=THR, backbone="dinov2_vits14", **out_all)
    (RUN / "results").mkdir(exist_ok=True)
    (RUN / "results" / "sun_audit.json").write_text(json.dumps(out, indent=1))
    print("[done] results/sun_audit.json")


if __name__ == "__main__":
    main()
