"""
Modal launcher: run_predicate_grounding_milestone.py on the CUB data volume.

Runs the review-milestone experiment (predicate agreement + correct/control
region perturbations) on a T4 GPU against the CUB-200-2011 volume, then
saves the JSON result into the repository's results/ directory locally.

Usage:
    python scripts/run_milestone_modal.py --train-limit 1500 --test-limit 500
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
PROJECT_DIR = "/root/grounded_vision"

app = modal.App("gv-milestone")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install("scikit-learn>=1.2", "pyyaml>=6", "pillow>=9", "tqdm")
    .add_local_dir(
        str(REPO),
        remote_path=PROJECT_DIR,
        ignore=lambda p: any(
            s in str(p) for s in (".git", "cache", "__pycache__", ".pytest_cache")
        ),
    )
)
data_volume = modal.Volume.from_name("grounded-vision-cub-data", create_if_missing=False)


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/data": data_volume})
def run(train_limit: int, test_limit: int, pos_train_min: int, pos_test_min: int) -> str:
    data_dir = "/data/CUB_200_2011"
    if not Path(data_dir, "attributes.txt").exists():
        data_dir = "/data/CUB_200_2011/CUB_200_2011"
    if not Path(data_dir, "attributes.txt").exists():
        raise FileNotFoundError("CUB attributes.txt not found on the volume")
    out = "/tmp/predicate_grounding_milestone.json"
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_DIR}/src"}
    subprocess.run(
        [
            "python", f"{PROJECT_DIR}/scripts/run_predicate_grounding_milestone.py",
            "--data-dir", data_dir,
            "--out", out,
            "--train-limit", str(train_limit),
            "--test-limit", str(test_limit),
            "--pos-train-min", str(pos_train_min),
            "--pos-test-min", str(pos_test_min),
        ],
        check=True,
        env=env,
        cwd=PROJECT_DIR,
    )
    return Path(out).read_text()


@app.local_entrypoint()
def main(
    train_limit: int = 1500,
    test_limit: int = 500,
    pos_train_min: int = 30,
    pos_test_min: int = 10,
):
    text = run.remote(train_limit, test_limit, pos_train_min, pos_test_min)
    out = REPO / "results" / "predicate_grounding_milestone.json"
    out.write_text(text)
    print(f"saved: {out}")
