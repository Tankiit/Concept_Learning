"""Shared plumbing for the three hand-off tasks (Tasks 1-3, Nov 11-16).

Ground rules enforced here (see README.md and the pre-registration):
  - pipeline runs write NEW result files; existing files are never overwritten
    (pass force=True only for deliberate, named reruns)
  - everything runs under the repo venv with PYTHONPATH=src
  - pass/fail lines print as "PASS:" / "FAIL:" so a CI or a human can grep
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The audit/X2 pipelines need torch+torchvision+sklearn: the anaconda python has
# the full stack; the repo venv (ltn_pyxai_starter) has pyxai but no torchvision.
PIPE_PY = "/Users/tanmoy/anaconda3/bin/python"
if not os.path.exists(PIPE_PY):
    PIPE_PY = os.path.join(REPO, "ltn_pyxai_starter", ".venv", "bin", "python")
DEFAULT_CACHE = "/Users/tanmoy/research/data/grounded_vision_runs/cub_local_et"
DEFAULT_DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"

PREREG = ("https://github.com/ (pre-registration: docs/wiki/"
          "CVPR2027_PREREG_TASK_RUNS.md in this repo)")


def out_path(name):
    p = os.path.join(REPO, "results", name)
    if os.path.exists(p):
        raise SystemExit(f"REFUSING to overwrite existing {p}; choose a new file name")
    return p


def run_pipeline(script, out, cache_dir, data_dir, extra=(), force=False):
    """Run a grounded_vision pipeline script under the venv. Returns the out path."""
    if os.path.exists(out) and not force:
        print(f"SKIP (exists, not overwriting): {out}")
        return out
    cmd = [PIPE_PY, os.path.join(REPO, "scripts", script),
           "--cache-dir", cache_dir, "--data-dir", data_dir, "--out", out, *extra]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    if not os.path.exists(out):
        raise SystemExit(f"pipeline did not write {out}")
    return out


def run_script(script, script_args, force=False):
    """Run a repo script (no cache/data injection) under the venv."""
    path = script if os.path.isabs(script) else os.path.join(REPO, "scripts", script)
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(REPO, "scripts", "tasks", script)
    cmd = [PIPE_PY, path, *script_args]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def check(label, ok):
    print(("PASS: " if ok else "FAIL: ") + label)
    return bool(ok)


def finish(checks):
    ok = all(checks)
    print("\n== TASK RESULT:", "PASS" if ok else "FAIL", "==")
    sys.exit(0 if ok else 1)


def n_pct(k, n):
    """Every percentage reported beside its n."""
    return f"{k}/{n} = {k / n:.3f}" if n else "0/0 = n/a"


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
