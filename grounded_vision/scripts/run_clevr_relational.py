"""
Script: run_clevr_relational.py
===============================
Certified relational track: answers to CLEVR-style existential questions
("is there a red cube left of a sphere?") over ground-truth scene graphs,
with verified sufficient reasons that cite RELATIONAL predicates.

Scene source: either a real CLEVR scenes.json (--scenes-json) or scenes
sampled from the documented CLEVR generative process (fallback; labeled
CLEVR-style synthetic). Only scene graphs are used — this is the oracle
grounding track (like --regions gt on CUB); no images or backbone needed.

Pipeline: scene graph -> FactDatabase (object entities, has_attribute facts,
spatial facts left_of/right_of/front_of/behind over all pairs) -> existential
concept/relational features -> validation-selected tree -> native certified
sufficient reasons + contrastives.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_vision.predicates.schema import Entity, Fact, FactDatabase, Provenance, PredicateType
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend

COLORS = ["gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan"]
SHAPES = ["cube", "sphere", "cylinder"]
SIZES = ["small", "large"]
MATERIALS = ["rubber", "metal"]
RELS = ["left_of", "right_of", "front_of", "behind"]
IMG_W, IMG_H = 480, 320
OUT = "/Users/cril/tanmoy/research/data/grounded_vision_runs/clevr_relational"


# ---------------------------------------------------------------- scenes ----
def sample_scene(rng) -> list:
    """CLEVR-style scene: objects with attrs and pixel coords (rejection
    sampling with size-dependent minimum distance, as in the CLEVR engine)."""
    n = rng.randint(3, 11)
    objs, positions = [], []
    for _ in range(n):
        for _try in range(60):
            x = rng.uniform(0.12, 0.88) * IMG_W
            y = rng.uniform(0.55, 0.95) * IMG_H
            size = rng.choice(SIZES)
            ok = all((x - px) ** 2 + (y - py) ** 2 >= (0.13 * IMG_W if "large" in (size, ps) else 0.10 * IMG_W) ** 2
                     for _, (px, py, ps) in positions)
            if ok:
                positions.append((None, (x, y, size)))
                objs.append({"shape": rng.choice(SHAPES), "size": size,
                             "color": rng.choice(COLORS), "material": rng.choice(MATERIALS),
                             "x": x, "y": y})
                break
    return objs


def load_real_scenes(path: str, limit: int):
    data = json.load(open(path))
    scenes = []
    for sc in data[:limit]:
        objs = []
        for o in sc["objects"]:
            px = o.get("pixel_coords") or [o["3d_coords"][0], o["3d_coords"][1]]
            objs.append({"shape": o["shape"], "size": o["size"], "color": o["color"],
                         "material": o["material"], "x": float(px[0]), "y": float(px[1])})
        scenes.append(objs)
    return scenes


# ------------------------------------------------------------- predicates ----
def scene_facts(objs) -> FactDatabase:
    db = FactDatabase(sample_id=0)
    ids = []
    for i, o in enumerate(objs):
        oid = f"obj_{i}"
        ids.append(oid)
        db.add_entity(Entity(id=oid, entity_type="object", metadata=o))
        for attr, val in (("color", o["color"]), ("shape", o["shape"]),
                          ("size", o["size"]), ("material", o["material"])):
            db.add_fact(Fact(predicate="has_attribute", subject_id=oid, attribute=attr,
                             value=val, confidence=1.0, truth_value=True,
                             provenance=None, predicate_type=PredicateType.HAS_ATTRIBUTE))
    for a in range(len(objs)):
        for b in range(len(objs)):
            if a == b:
                continue
            oa, ob = objs[a], objs[b]
            holds = None
            if oa["x"] < ob["x"]:
                holds = "left_of"
            elif oa["x"] > ob["x"]:
                holds = "right_of"
            if oa["y"] > ob["y"]:
                holds2 = "front_of"
            elif oa["y"] < ob["y"]:
                holds2 = "behind"
            else:
                holds2 = None
            for rel in (holds, holds2):
                if rel:
                    db.add_fact(Fact(predicate=rel, subject_id=ids[a], attribute=None,
                                     value=ids[b], confidence=1.0, truth_value=True,
                                     provenance=None, predicate_type=PredicateType.SPATIAL))
    return db


# -------------------------------------------------------------- questions ----
def gen_question(rng, objs):
    """Balanced existential relational question, always yields. A-spec comes
    from an object present in the scene; half the questions are constructed
    true and half verified false, so labels stay ~50/50."""
    if len(objs) < 2:
        return None
    def spec_of(o, kind):
        if kind == "color_shape":
            return [("color", str(o["color"])), ("shape", str(o["shape"]))]
        if kind == "shape":
            return [("shape", str(o["shape"]))]
        return [("color", str(o["color"]))]
    want_true = rng.rand() < 0.5
    for _ in range(12):
        ia, ib = rng.choice(len(objs), size=2, replace=False)
        a_spec = spec_of(objs[ia], "color_shape")
        b_spec = spec_of(objs[ib], "shape")
        true_rels = [r for r in RELS if rel_holds(objs[ia], objs[ib], r)]
        false_rels = [r for r in RELS if not rel_holds(objs[ia], objs[ib], r)]
        if want_true and true_rels:
            rel = true_rels[rng.randint(len(true_rels))]
        elif not want_true and false_rels:
            rel = false_rels[rng.randint(len(false_rels))]
        else:
            continue
        # Label = EXISTENTIAL truth over the scene (not the sampled pair):
        # the question's semantics is "does ANY matching pair exist".
        def matches(o, spec):
            return all(o[k] == v for k, v in spec)
        ans = int(any(matches(objs[a], a_spec) and matches(objs[b], b_spec)
                      and rel_holds(objs[a], objs[b], rel)
                      for a in range(len(objs)) for b in range(len(objs)) if a != b))
        return a_spec, b_spec, rel, ans
    return None


def rel_holds(a, b, rel):
    if rel == "left_of":
        return a["x"] < b["x"]
    if rel == "right_of":
        return a["x"] > b["x"]
    if rel == "front_of":
        return a["y"] > b["y"]
    return a["y"] < b["y"]


# --------------------------------------------------------------- features ----
def feature_names():
    names = []
    for c in COLORS:
        names.append(f"exists(color={c})")
    for s in SHAPES:
        names.append(f"exists(shape={s})")
    for z in SIZES:
        names.append(f"exists(size={z})")
    for m in MATERIALS:
        names.append(f"exists(material={m})")
    for c in COLORS:
        for s in SHAPES:
            names.append(f"exists(color={c};shape={s})")
    for m in MATERIALS:
        for s in SHAPES:
            names.append(f"exists(material={m};shape={s})")
    for rel in RELS:
        for c1 in COLORS:
            for s1 in SHAPES:
                for s2 in SHAPES:
                    names.append(f"exists({rel}(color={c1};shape={s1},shape={s2}))")
    return names


QUESTION_FEATURES = ([f"q.A.color={c}" for c in COLORS]
                     + [f"q.A.shape={sh}" for sh in SHAPES]
                     + [f"q.B.color={c}" for c in COLORS]
                     + [f"q.B.shape={sh}" for sh in SHAPES]
                     + [f"q.rel={r}" for r in RELS])
FEATURES = feature_names() + QUESTION_FEATURES
FIDX = {n: i for i, n in enumerate(FEATURES)}
REL_FEATURES = [i for i, n in enumerate(FEATURES) if "(" in n]


def scene_features(objs, q=None) -> np.ndarray:
    x = np.zeros(len(FEATURES), dtype=np.int64)
    for o in objs:
        x[FIDX[f"exists(color={o['color']})"]] = 1
        x[FIDX[f"exists(shape={o['shape']})"]] = 1
        x[FIDX[f"exists(size={o['size']})"]] = 1
        x[FIDX[f"exists(material={o['material']})"]] = 1
        x[FIDX[f"exists(color={o['color']};shape={o['shape']})"]] = 1
        x[FIDX[f"exists(material={o['material']};shape={o['shape']})"]] = 1
    for a in objs:
        for b in objs:
            if a is b:
                continue
            for rel in RELS:
                if rel_holds(a, b, rel):
                    x[FIDX[f"exists({rel}(color={a['color']};shape={a['shape']},"
                           f"shape={b['shape']}))"]] = 1
    if q is not None:
        a_spec, b_spec, rel, _ = q
        for k, v in a_spec:
            x[FIDX[f"q.A.{k}={v}"]] = 1
        for k, v in b_spec:
            x[FIDX[f"q.B.{k}={v}"]] = 1
        x[FIDX[f"q.rel={rel}"]] = 1
    return x


# ------------------------------------------------------------------- main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--scenes-json", type=str, default=None,
                        help="real CLEVR scenes.json if available")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rng = np.random.RandomState(42)
    source = "clevr_style_synthetic"
    if args.scenes_json:
        scenes = load_real_scenes(args.scenes_json, args.n_train + args.n_test)
        source = f"real:{os.path.basename(args.scenes_json)}"
        print(f"--> loaded {len(scenes)} real scenes")
    train_scenes = [sample_scene(rng) for _ in range(args.n_train)] if not args.scenes_json else scenes[:args.n_train]
    test_scenes = [sample_scene(rng) for _ in range(args.n_test)] if not args.scenes_json else scenes[args.n_train:args.n_train + args.n_test]
    print(f"--> source: {source} | train {len(train_scenes)} / test {len(test_scenes)} scenes")

    def build(scenes):
        X, y, qs = [], [], []
        for si, objs in enumerate(scenes):
            q = gen_question(rng, objs)
            if q is None or len(objs) < 2:
                continue
            a_spec, b_spec, rel, ans = q
            X.append(scene_features(objs, q))
            y.append(ans)
            qs.append({"scene": si, "A": a_spec, "B": b_spec, "rel": rel, "answer": ans})
        return np.array(X), np.array(y), qs

    Xtr, ytr, qtr = build(train_scenes)
    Xte, yte, qte = build(test_scenes)
    print(f"questions: train {len(ytr)} (pos {ytr.mean():.2f}) | test {len(yte)} (pos {yte.mean():.2f})")

    # ---- validation-selected tree -------------------------------------------
    from sklearn.tree import DecisionTreeClassifier
    n = len(Xtr)
    perm = rng.permutation(n)
    n_val = int(0.15 * n)
    vi, fi = perm[:n_val], perm[n_val:]
    candidates = [("d8", {"max_depth": 8}), ("d12", {"max_depth": 12}),
                  ("d20", {"max_depth": 20}), ("msl_8", {"min_samples_leaf": 8}),
                  ("msl_32", {"min_samples_leaf": 32}),
                  ("leaves_64", {"max_leaf_nodes": 64}),
                  ("leaves_256", {"max_leaf_nodes": 256})]
    best, best_val = None, -1
    for name, params in candidates:
        dt = DecisionTreeClassifier(random_state=42, **params).fit(Xtr[fi], ytr[fi])
        val = float(np.mean(dt.predict(Xtr[vi]) == ytr[vi]))
        if val > best_val:
            best, best_val = (name, params), val
    print(f"--> selected {best[0]} on validation (val acc {best_val*100:.2f}%)")
    dt = DecisionTreeClassifier(random_state=42, **best[1]).fit(Xtr, ytr)
    head = CertifiableTreeHead(feature_names=FEATURES, class_names=["no", "yes"])
    head.clf, head.is_fitted = dt, True
    preds = dt.predict(Xte)
    test_acc = float(np.mean(preds == yte))
    print(f"--> TEST accuracy: {test_acc*100:.2f}%")

    # ---- certified reasons ---------------------------------------------------
    backend = PyXAIBackend(head)
    rows = rng.choice(len(Xte), size=min(200, len(Xte)), replace=False)
    sizes, rel_counts, examples = [], [], []
    conds_with_rel = 0
    for i in rows:
        x = Xte[i]
        pred = int(preds[i])
        exp = backend.explain_sufficient(x, pred, sample_id=int(i))
        names = [FEATURES[c.feature_idx] for c in exp.conditions]
        n_rel = sum(1 for c in exp.conditions if c.feature_idx in REL_FEATURES)
        conds_with_rel += 1 if n_rel else 0
        sizes.append(len(exp.conditions))
        rel_counts.append(n_rel)
        if len(examples) < 4:
            con = backend.explain_contrastive(x, pred, sample_id=int(i))
            examples.append({
                "question": qte[int(i)], "prediction": pred,
                "reason": [{"condition": n, "op": c.operator, "thr": float(c.threshold)}
                           for n, c in zip(names, exp.conditions)],
                "contrastive": [c.condition_str for c in con.conditions_to_alter] if con else None,
            })
    necessity = TreeSufficiencyNecessity(head)
    print(f"reason size: mean {np.mean(sizes):.2f} max {max(sizes)} | "
          f"reasons citing >=1 relational literal: {conds_with_rel}/{len(rows)} "
          f"(mean relational literals {np.mean(rel_counts):.2f})")

    report = {
        "source": source, "selected_tree": best[0], "test_acc": test_acc,
        "n_questions": {"train": len(ytr), "test": len(yte)},
        "positive_rate": float(yte.mean()),
        "reason_mean": float(np.mean(sizes)), "reason_max": int(max(sizes)),
        "frac_reasons_with_relational_literal": conds_with_rel / len(rows),
        "mean_relational_literals_per_reason": float(np.mean(rel_counts)),
        "necessity_rate": necessity(head, Xte, preds, rows),
        "examples": examples,
    }
    with open(f"{OUT}/clevr_relational_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"saved: {OUT}/clevr_relational_results.json")
    for e in examples[:3]:
        print(" Q:", e["question"]["A"], e["question"]["rel"], e["question"]["B"],
              "| ans", e["question"]["answer"], "pred", e["prediction"])
        print("   reason:", " AND ".join(
            f"[{'NOT ' if c['op']=='<=' else ''}{c['condition']}]" for c in e["reason"]))


def TreeSufficiencyNecessity(head):
    from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier
    from grounded_vision.explanations.schema import ExplanationCondition, VerificationStatus
    verifier = TreeSufficiencyVerifier(head.clf, head.feature_names)
    backend = PyXAIBackend(head)

    def check(head_, Xte, preds, rows):
        n_ok, n = 0, 0
        for i in rows[:100]:
            exp = backend.explain_sufficient(Xte[i], int(preds[i]), sample_id=int(i))
            for j in range(len(exp.conditions)):
                subset = [c for k, c in enumerate(exp.conditions) if k != j]
                conds = [ExplanationCondition(c.feature_idx, c.feature_name, c.operator,
                                              c.threshold, observed_value=0.0) for c in subset]
                n += 1
                if verifier.verify_sufficiency(conds, int(preds[i])).status != VerificationStatus.VERIFIED:
                    n_ok += 1
        return n_ok / max(1, n)
    return check


if __name__ == "__main__":
    main()
