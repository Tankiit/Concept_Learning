"""
Script: compare_patch_transform.py
==================================
Raw vs rotated vs whitened patch embeddings, end to end, on the same feature cache.

One isolated run directory per variant, {out_dir}/{raw,rotated,whitened}[_seedN]/, whose
{train,test}_features.npz are SYMLINKS to the base cache (no duplicated GBs);
each runs the real pipeline scripts:

    fit_perception.py --no-rotate | --rotate | --whiten   (readers + prototypes)
    train_head.py                              (tree on the new predicates)
    evaluate.py                                (perception / species / explanations)
    analyze_concept_correctness.py             (are certified clauses true of the image?)
    analyze_concept_ceilings.py                (optional: logreg on the same concepts)

and writes one comparison table ({out_dir}/patch_transform_comparison.{json,md}).
Everything except the patch transform is held fixed (same cache, seeds, epochs,
attr-weight, tree hyperparameters). Use --seeds 42,43,44 to repeat every
variant under several seeds (mean ± std reported).

Built-in sanity check: the PartDetector reads the raw CLS token, so part
IoU must be identical between variants under the same seed.

CAVEAT: all metrics are on the TEST split -- report them as an effect,
not as a tuned final number.

Usage:
    python scripts/compare_patch_transform.py --config configs/cub_bound_local.yaml \
        --epochs 200 --attr-weight 1.0 --ceilings
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VARIANTS = ("raw", "rotated", "whitened")
FIT_FLAGS = {"raw": ["--no-rotate", "--no-whiten"],
             "rotated": ["--rotate", "--no-whiten"],
             "whitened": ["--whiten"]}

# (label, source, dotted key, higher_is_better) -- source names the JSON file below
METRICS = [
    ("Attr. accuracy (train)",        "fit",  "attribute_accuracy_train",                   True),
    ("Attr. accuracy (test)",         "eval", "perception.attribute_accuracy",              True),
    ("Attr. F1 (test)",               "eval", "perception.attribute_f1",                    True),
    ("Attr. precision (test)",        "eval", "perception.attribute_precision",             True),
    ("Attr. recall (test)",           "eval", "perception.attribute_recall",                True),
    ("Part IoU (sanity: constant)",   "eval", "perception.mean_part_iou",                   None),
    ("Species accuracy (tree)",       "eval", "classification.accuracy",                    True),
    ("Species macro-F1 (tree)",       "eval", "classification.macro_f1",                    True),
    ("Species acc. (logreg ceiling)", "ceil", "logreg_on_predicted_concepts.test_accuracy", True),
    ("Sufficiency rate",              "eval", "explanations.sufficiency_rate",              True),
    ("Mean reason size",              "eval", "explanations.mean_sparsity_conditions",      False),
    ("NL faithful rate",              "eval", "explanations.nl_faithful_rate",              True),
    ("Clause correctness",            "corr", "clause_correctness.overall",                 True),
    ("Present-clause precision",      "corr", "clause_correctness.predicted_present",       True),
    ("Absent-clause accuracy",        "corr", "clause_correctness.predicted_absent",        True),
    ("P(all clauses correct)",        "corr", "p_all_clauses_correct_given_reason",         True),
]
PCT_EXCLUDE = {"Mean reason size"}


def dig(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def run(cmd, log_path):
    """Runs one pipeline step, appending its output to the variant log."""
    print("   $", " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT + os.sep) else c
                           for c in cmd))
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"step failed (exit {proc.returncode}): {' '.join(cmd)} -- see {log_path}")
    return time.time() - t0


def prepare_variant(cfg, base_cache, vdir, variant, seed):
    os.makedirs(vdir, exist_ok=True)
    for split in ("train", "test"):
        src = os.path.abspath(os.path.join(base_cache, f"{split}_features.npz"))
        dst = os.path.join(vdir, f"{split}_features.npz")
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} missing -- run cache_features.py --split all --eval-transform")
        if os.path.lexists(dst):
            os.remove(dst)  # only ever a symlink we created
        os.symlink(src, dst)
    vcfg = json.loads(json.dumps(cfg))
    vcfg["data"]["cache_dir"] = vdir
    vcfg["perception"]["rotate_patches"] = variant != "raw"
    vcfg["perception"]["whiten_patches"] = variant == "whitened"
    vcfg.setdefault("experiment", {})["seed"] = seed
    cfg_path = os.path.join(vdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(vcfg, f, sort_keys=False)
    return cfg_path


def collect(vdir, regions):
    return {
        "fit": load_json(os.path.join(vdir, "perception_train_metrics.json")),
        "eval": load_json(os.path.join(vdir, "results", "benchmark_results.json")),
        "corr": load_json(os.path.join(vdir, f"concept_correctness_{regions}_regions.json")),
        "ceil": load_json(os.path.join(vdir, f"concept_ceilings_{regions}_regions.json")),
    }


def fmt(label, vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "—", None
    m = float(np.mean(vals))
    pct = label not in PCT_EXCLUDE
    s = f"{100 * m:.2f}%" if pct else f"{m:.2f}"
    if len(vals) > 1:
        sd = float(np.std(vals, ddof=1))
        s += f" ± {100 * sd:.2f}" if pct else f" ± {sd:.2f}"
    return s, m


def build_table(results, seeds, variants):
    others = [v for v in variants if v != "raw"]
    lines = ["| Metric | " + " | ".join(variants) + " | " + " | ".join(f"Δ {v}" for v in others) + " |",
             "|---|" + "---:|" * (len(variants) + len(others))]
    for label, src, key, hib in METRICS:
        cells, means = [], {}
        for v in variants:
            s, m = fmt(label, [dig(results[(v, sd)][src], key) for sd in seeds])
            cells.append(s)
            means[v] = m
        deltas = []
        for v in others:
            delta = ""
            if means.get("raw") is not None and means[v] is not None:
                d = means[v] - means["raw"]
                pct = label not in PCT_EXCLUDE
                delta = f"{100 * d:+.2f} pt" if pct else f"{d:+.2f}"
                if hib is not None and abs(d) > 1e-9:
                    delta += " ▲" if (d > 0) == hib else " ▼"
            deltas.append(delta)
        lines.append(f"| {label} | " + " | ".join(cells + deltas) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compare raw vs rotated vs whitened patch embeddings.")
    ap.add_argument("--config", default="configs/cub_bound_local.yaml")
    ap.add_argument("--out-dir", default=None, help="default: {cache_dir}/patch_transform_comparison")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,43,44")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--attr-weight", type=float, default=1.0)
    ap.add_argument("--regions", default="gt", choices=["gt", "predicted"])
    ap.add_argument("--num-samples", type=int, default=0, help="test samples for evaluate.py (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-explanations", action="store_true")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--ceilings", action="store_true", help="also run the logreg ceiling")
    ap.add_argument("--reuse", action="store_true",
                    help="skip runs that already have a benchmark_results.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    base_cache = cfg["data"]["cache_dir"]
    if not os.path.isabs(base_cache):
        base_cache = os.path.join(ROOT, base_cache)
    out_dir = os.path.abspath(args.out_dir or os.path.join(base_cache, "patch_transform_comparison"))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = list(VARIANTS)

    py = sys.executable
    S = lambda name: os.path.join(HERE, name)  # noqa: E731
    results = {}
    for v in variants:  # variant-major: each variant finishes all seeds before the next
        for seed in seeds:
            name = v if len(seeds) == 1 else f"{v}_seed{seed}"
            vdir = os.path.join(out_dir, name)
            print(f"\n=== {name} ===")
            vcfg = prepare_variant(cfg, base_cache, vdir, v, seed)
            if args.reuse and os.path.exists(os.path.join(vdir, "results", "benchmark_results.json")):
                print("   (reusing existing results)")
                results[(v, seed)] = collect(vdir, args.regions)
                continue
            log = os.path.join(vdir, "run.log")
            open(log, "w").close()
            t0 = time.time()
            run([py, S("fit_perception.py"), "--config", vcfg, *FIT_FLAGS[v],
                 "--epochs", str(args.epochs), "--attr-weight", str(args.attr_weight),
                 "--device", args.device], log)
            run([py, S("train_head.py"), "--config", vcfg, "--regions", args.regions], log)
            ev = [py, S("evaluate.py"), "--config", vcfg, "--regions", args.regions,
                  "--num-samples", str(args.num_samples)]
            if args.skip_explanations:
                ev.append("--skip-explanations")
            run(ev, log)
            if not args.skip_correctness:
                cc = [py, S("analyze_concept_correctness.py"), "--config", vcfg,
                      "--regions", args.regions]
                if args.num_samples:
                    cc += ["--limit", str(args.num_samples)]
                run(cc, log)
            if args.ceilings:
                try:  # full-split only (no --limit); non-fatal so the table still comes out
                    run([py, S("analyze_concept_ceilings.py"), "--config", vcfg,
                         "--regions", args.regions], log)
                except RuntimeError as e:
                    print(f"   WARNING: ceilings skipped -- {e}")
            results[(v, seed)] = collect(vdir, args.regions)
            print(f"   done in {time.time() - t0:.0f}s -- log: {log}")

    def _iou(v, s):
        return dig(results[(v, s)]["eval"], "perception.mean_part_iou")
    iou_ok = all(_iou("raw", s) is None or _iou(v, s) is None or abs(_iou("raw", s) - _iou(v, s)) < 1e-4
                 for s in seeds for v in variants)
    header = (f"# Patch transform comparison (raw / rotated / whitened)\n\nConfig: `{args.config}` · regions: `{args.regions}` · "
              f"epochs {args.epochs} · attr-weight {args.attr_weight} · seeds {seeds} · "
              f"test samples: {args.num_samples or 'all'}\n\n"
              "Δ = variant − raw (mean over seeds; ± is the std across seeds); ▲ better, ▼ worse. "
              "TEST split numbers.\n\n")
    sanity = ("Part IoU identical between variants ✔ (only the patch transform changed)\n" if iou_ok else
              "⚠ Part IoU differs between variants — something besides the patch transform changed!\n")
    md = header + build_table(results, seeds, variants) + "\n\n" + sanity
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "patch_transform_comparison.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, "patch_transform_comparison.json"), "w") as f:
        json.dump({"config": args.config, "seeds": seeds, "variants": variants, "args": vars(args),
                   "part_iou_constant": iou_ok,
                   "runs": {f"{v}_seed{s}": r for (v, s), r in results.items()}}, f, indent=2)
    print("\n" + md)
    print(f"saved: {os.path.join(out_dir, 'patch_transform_comparison.md')} (+ .json)")


if __name__ == "__main__":
    main()
