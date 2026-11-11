"""
test_run_experiment.py — integration test for the bankexp pipeline.

Ported in structure from CSU-TrustLab/behavior-explainer
tests/test_run_experiment.py: unit tests for the shared CSV/reasons
formats always run; the integration test runs the pipeline end-to-end
on a small SYNTHETIC bank (teacher -> distill -> explain --native-only
-> metric 1) and asserts the output files exist and satisfy the schema
invariants.  PyXAI is not required (native verifier path), so the test
runs in any environment with sklearn + numpy.

Binary reason format recap
--------------------------
reasons_K{K}.jsonl: one JSON record per test image with REQUIRED keys
    sample_id, K, pred_tree, native_reason, native_size,
    native_verified, status
where native_reason is a list of [feature_idx+1, bit] pairs (signed
1-based feature indices in the pyxai convention when present).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC = PROJECT_ROOT / "src"
ANALYSIS = PROJECT_ROOT / "analysis"

REASON_REQUIRED_KEYS = ["sample_id", "K", "pred_tree", "native_reason",
                        "native_size", "native_verified", "status"]
METRIC1_COLUMNS = ["method", "K", "sample_id", "size",
                   "sufficient_exact", "worst_margin", "n_classes_checked"]


# ---------------------------------------------------------------------------
# Unit tests — format invariants over any existing artifacts
# ---------------------------------------------------------------------------

def parse_reasons(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_metric_csv(path: Path):
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    header = lines[0].split(",")
    rows = [dict(zip(header, l.split(","))) for l in lines[1:]]
    return header, rows


def test_reasons_format():
    path = PROJECT_ROOT / "results" / "reasons_K4.jsonl"
    if not path.is_file():
        pytest.skip("results/reasons_K4.jsonl not found (run the pipeline)")
    for rec in parse_reasons(path):
        assert all(k in rec for k in REASON_REQUIRED_KEYS), \
            f"record missing keys: {sorted(set(REASON_REQUIRED_KEYS) - set(rec))}"
        assert rec["status"] in ("ok",) or rec["status"].startswith("error")
        if rec["status"] == "ok":
            assert rec["native_size"] == len(rec["native_reason"])


def test_metric1_format():
    path = PROJECT_ROOT / "results" / "metric1_margin_sufficiency_K4.csv"
    if not path.is_file():
        pytest.skip("metric1 CSV not found (run the pipeline)")
    header, rows = parse_metric_csv(path)
    assert header == METRIC1_COLUMNS
    assert rows, "metric1 CSV should have data rows"
    for r in rows:
        int(r["size"]); int(r["sufficient_exact"]); float(r["worst_margin"])


# ---------------------------------------------------------------------------
# Integration test — synthetic bank, full chain, no PyXAI
# ---------------------------------------------------------------------------

def make_synthetic_bank(out_dir: Path, n_train=120, n_test=40, F=8,
                        n_classes=3, seed=7):
    """Linear-separable-ish random bits so the tree has real structure."""
    rng = np.random.RandomState(seed)
    protos = rng.randint(0, 2, size=(n_classes, F))
    packs = []
    for split, n, base in (("train", n_train, 0), ("test", n_test, 1000)):
        y = rng.randint(0, n_classes, size=n)
        noise = rng.rand(n, F) < 0.15
        X = (protos[y] ^ noise).astype(np.int8)
        packs.append(dict(
            sample_ids=np.arange(base, base + n, dtype=np.int64),
            labels=y.astype(np.int64), bits=X,
            scores=X.astype(np.float32), cal=X.astype(np.float32),
            gt=np.zeros_like(X)))
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, p in zip(("train", "test"), packs):
        np.savez(out_dir / f"bank_{split}.npz", **p)
    (out_dir / "meta.json").write_text(json.dumps({
        "feature_names": [f"f{i}" for i in range(F)],
        "feat_part": [""] * F, "part_feat_idx": [], "vocab": [],
        "protocol": {"synthetic": True},
    }))
    return out_dir


def test_pipeline_end_to_end(tmp_path):
    if shutil.which("python") is None and not Path(sys.executable).is_file():
        pytest.skip("no interpreter")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        pytest.skip("scikit-learn not installed")
    bank = make_synthetic_bank(tmp_path / "bank")
    K = 3
    def run(mod, *extra):
        proc = subprocess.run(
            [sys.executable, str(mod), *map(str, extra)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc
    run(SRC / "teacher.py", "--bank", bank, "--classes", K,
        "--out", tmp_path)
    run(SRC / "distill.py", "--bank", bank, "--classes", K,
        "--teacher", tmp_path / f"teacher_K{K}.npz", "--out", tmp_path)
    run(SRC / "explain.py", "--bank", bank,
        "--tree", tmp_path / f"tree_K{K}.json", "--classes", K,
        "--out", tmp_path / "reasons.jsonl", "--native-only")
    run(ANALYSIS / "margin_sufficiency.py", "--bank", bank,
        "--teacher", tmp_path / f"teacher_K{K}.npz",
        "--reasons", tmp_path / "reasons.jsonl",
        "--out", tmp_path / "metric1.csv")

    recs = parse_reasons(tmp_path / "reasons.jsonl")
    assert len(recs) == 40
    for rec in recs:
        assert rec["status"] == "ok"
        assert rec["native_verified"] is True  # tree reasons are exact
    header, rows = parse_metric_csv(tmp_path / "metric1.csv")
    assert header == METRIC1_COLUMNS
    assert len(rows) == 40
    assert {r["method"] for r in rows} == {"native"}


if __name__ == "__main__":
    test_reasons_format()
    test_metric1_format()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_pipeline_end_to_end(Path(td))
    print("All tests passed!")
