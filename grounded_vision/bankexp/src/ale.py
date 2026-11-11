"""
ale.py — thin wrapper that calls beyond_l2's run_formal_exp.py.
================================================================
Stage 5a (any env).  The ONLY module allowed to leave our stack: it
invokes the external beyond_l2 repository as a subprocess.  The
wrapper is honest about absence: if the repo is not on disk it reports
what it needs and exits 3 (no synthetic results, ever).  No copy of
beyond_l2 exists in this workspace at build time (searched 2026-11-08);
the interface below is the contract for when it is cloned.

Usage
-----
    python src/ale.py --repo /path/to/beyond_l2 --classes 4
    python src/ale.py --check --repo /path/to/beyond_l2

Exit codes: 0 ok, 3 repo missing, 4 their script failed.
"""

from __future__ import annotations
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
RESULTS_DIR = PROJECT_ROOT / "results"

REQUIRED_FILES = ["run_formal_exp.py"]


def export_bank_csv(bank_dir, path, classes_k):
    te = np.load(Path(bank_dir) / "bank_test.npz")
    mask = te["labels"] < classes_k
    ids, X, y = te["sample_ids"][mask], te["bits"][mask], te["labels"][mask]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "label"] + [f"f{i}" for i in range(X.shape[1])])
        for i in range(len(ids)):
            w.writerow([int(ids[i]), int(y[i])] + [int(v) for v in X[i]])
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("BEYOND_L2_REPO", ""))
    ap.add_argument("--bank", default=str(INTERMEDIATE_DIR))
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true",
                    help="only verify the repo is usable, then exit")
    args = ap.parse_args()
    out = args.out or str(RESULTS_DIR / f"metric5_ale_K{args.classes}.csv")

    if not args.repo:
        print("ERROR: no --repo and $BEYOND_L2_REPO is unset. Clone the "
              "beyond_l2 repository (the ALE formal-explanation baseline) "
              "and pass its checkout path.", file=sys.stderr)
        sys.exit(3)
    repo = Path(args.repo).expanduser().resolve()
    missing = [f for f in REQUIRED_FILES if not (repo / f).exists()]
    if missing:
        print(f"ERROR: {repo} is missing {missing}; not a usable beyond_l2 "
              f"checkout.", file=sys.stderr)
        sys.exit(3)
    if args.check:
        print(f"OK: {repo}/run_formal_exp.py present")
        sys.exit(0)

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = export_bank_csv(args.bank, Path(out).with_suffix(".input.csv"),
                              args.classes)
    cmd = [sys.executable, str(repo / "run_formal_exp.py"),
           "--data", str(payload)]
    print(f"--> calling {cmd}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        print(f"ERROR: run_formal_exp.py exited {proc.returncode}",
              file=sys.stderr)
        sys.exit(4)
    # Their output format is not pinned in this workspace: save the raw
    # stdout and fail loudly rather than guessing a parse.
    Path(out).with_suffix(".their_stdout.txt").write_text(proc.stdout)
    print("NOTE: their stdout saved next to the output; mapping into the "
          "shared schema requires their output format -- extend "
          "parse_their_output() after cloning the repo.", file=sys.stderr)
    sys.exit(4)


if __name__ == "__main__":
    main()
