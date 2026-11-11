"""
run_experiment.py — parametrized experiment runner.

Runs the full pipeline for one class count K:

    1. bank      (conda env)     DINO cache -> binary concept bank
                                [skipped if cached, unless --rebuild-bank]
    2. teacher   (conda env)     linear head over the bank bits
    3. distill   (PyXAI env)     CART on teacher predictions
    4. explain   (PyXAI env)     PyXAI reasons + independent verifier
    5. analysis  (any env)       metrics 1-3 + 6 (margin / coverage /
                                robustness / MHS max-coverage)
    6. conaxp    (any env)       metric 4 (erasure baseline), both orders
    7. summarize (any env)       results/summary.md over all CSVs

--bank-dir selects which bank to run on (the semantic predicate bank in
intermediate_results/, or an alternative feature bank built by
src/altbank.py in intermediate_results/bank_cls/ etc.); --tag suffixes
every results filename so banks can be compared side by side.

The two interpreters mirror the environment split:
    --conda-python   torch + sklearn + grounded_vision src   (default: python3)
    --pyxai-python   pyxai 2.0.1 + sklearn + numpy           (default:
                     ../../ltn_pyxai_starter/.venv/bin/python relative
                     to this project)

Usage
-----
    python src/run_experiment.py --classes 4 \
        --cache-dir <dino cache> --data-dir <CUB root>

Notes
-----
- ALE (metric 5) is NOT run here: the beyond_l2 repo is external;
  invoke src/ale.py once it is cloned (see src/ale.py docstring).
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
ANALYSIS = PROJECT_ROOT / "analysis"
DEFAULT_PYXAI = (PROJECT_ROOT.parent / "ltn_pyxai_starter"
                 / ".venv" / "bin" / "python")


def run(cmd, env_note):
    print(f"\n==> [{env_note}] {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--bank-dir", default=None,
                    help="bank directory (default: intermediate_results; "
                         "alt banks: intermediate_results/bank_cls etc.)")
    ap.add_argument("--tag", default="",
                    help="filename suffix for all results (e.g. _cls)")
    ap.add_argument("--cache-dir", default=None,
                    help="DINO feature cache (required to fit the bank)")
    ap.add_argument("--data-dir", default=None,
                    help="CUB root (required to fit the predicate bank)")
    ap.add_argument("--teacher-on", choices=["bits", "scores"], default="bits")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--conda-python", default=sys.executable)
    ap.add_argument("--pyxai-python", default=str(DEFAULT_PYXAI))
    ap.add_argument("--rebuild-bank", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    K = args.classes
    bank = Path(args.bank_dir) if args.bank_dir else (
        PROJECT_ROOT / "intermediate_results")
    tag = args.tag
    R = PROJECT_ROOT / "results"

    bank_cached = (bank / "bank_train.npz").exists()
    if args.rebuild_bank or not bank_cached:
        if not (args.cache_dir and args.data_dir):
            sys.exit("bank fit requires --cache-dir and --data-dir "
                     "(or let it load from cache)")
        cmd = [args.conda_python, SRC / "bank.py",
               "--cache-dir", args.cache_dir, "--data-dir", args.data_dir,
               "--out", bank]
        if args.rebuild_bank:
            cmd.append("--rebuild")
        run(cmd, "conda: bank")

    run([args.conda_python, SRC / "teacher.py",
         "--bank", bank, "--classes", K, "--on", args.teacher_on,
         "--out", bank], "conda: teacher")
    run([args.pyxai_python, SRC / "distill.py",
         "--bank", bank, "--classes", K, "--max-depth", args.max_depth,
         "--out", bank], "pyxai: distill")
    explain_cmd = [args.pyxai_python, SRC / "explain.py", "--bank", bank,
                   "--classes", K, "--out", R / f"reasons_K{K}{tag}.jsonl"]
    if args.limit:
        explain_cmd += ["--limit", args.limit]
    run(explain_cmd, "pyxai: explain")

    metric_num = {"margin_sufficiency": 1, "coverage": 2, "robustness": 3}
    for script, num in metric_num.items():
        run([args.pyxai_python, ANALYSIS / f"{script}.py",
             "--bank", bank, "--classes", K,
             "--reasons", R / f"reasons_K{K}{tag}.jsonl",
             "--out", R / f"metric{num}_{script}_K{K}{tag}.csv"],
            f"analysis: {script}")
    run([args.pyxai_python, ANALYSIS / "max_coverage.py",
         "--classes", K, "--reasons", R / f"reasons_K{K}{tag}.jsonl",
         "--out", R / f"metric6_max_coverage_K{K}{tag}.csv"],
        "analysis: max_coverage")
    for order in ("random", "contribution"):
        run([args.pyxai_python, SRC / "conaxp.py",
             "--bank", bank, "--classes", K, "--order", order,
             "--out", R / f"metric4_erasure_conaxp_K{K}{tag}_{order}.csv"],
            f"baseline: conaxp/{order}")
    run([args.pyxai_python, ANALYSIS / "summarize.py"], "analysis: summarize")
    print("\n==> pipeline complete; see results/summary.md", flush=True)


if __name__ == "__main__":
    main()
