import subprocess
from pathlib import Path

import modal

APP_NAME = "grounded-vision-cub-full"
PROJECT_DIR = "/root/grounded_vision"
DATA_DIR = "/data/CUB_200_2011"
RESULTS_DIR = "/results/cub_bound_real_full"

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install("scikit-learn>=1.2", "pyyaml>=6", "pillow>=9", "matplotlib>=3.6", "tqdm")
    .add_local_dir(
        "/Users/cril/tanmoy/research/Natural Language Explanation/grounded_vision",
        remote_path=PROJECT_DIR,
        ignore=lambda p: ".git" in str(p) or "cache" in str(p) or "__pycache__" in str(p),
    )
)
data_volume = modal.Volume.from_name("grounded-vision-cub-data")
results_volume = modal.Volume.from_name("grounded-vision-cub-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    volumes={"/data": data_volume, "/results": results_volume},
)
def run_pipeline():
    config = Path("/tmp/cub_bound_real_full.yaml")
    config.write_text(
        f'''experiment:
  name: "cub_bound_real_full"
  representation: "bound"
  seed: 42
data:
  dataset: "cub_200_2011"
  data_dir: "{DATA_DIR}"
  cache_dir: "{RESULTS_DIR}"
  image_size: 224
  num_classes: 200
  num_parts: 15
  num_attributes: 312
  use_mock_if_missing: false
perception:
  backbone: "dinov2_vits14"
  embedding_dim: 384
  patch_size: 14
  part_detection: true
  num_prototypes_per_attribute: 5
  roi_pool_size: 7
predicates:
  representation_mode: "bound"
  probability_threshold: 0.5
  include_part_binding: true
  include_relations: false
  track_provenance: true
reasoning:
  model_type: "decision_tree"
  max_depth: 12
  min_samples_leaf: 4
  criterion: "gini"
  enforce_domain_constraints: true
explanations:
  backend: "tree_verifier"
  max_conditions: 6
  find_contrastive: true
  evidence_crops: true
language:
  template_style: "scientific"
  decoder_model: "rule_based"
  strict_faithfulness_audit: true
'''
    )
    commands = [
        ["python", "scripts/cache_features.py", "--config", str(config)],
        ["python", "scripts/fit_perception.py", "--config", str(config)],
        ["python", "scripts/export_predicates.py", "--config", str(config)],
        ["python", "scripts/train_head.py", "--config", str(config)],
        ["python", "scripts/explain.py", "--config", str(config), "--sample-idx", "0"],
        ["python", "scripts/evaluate.py", "--config", str(config)],
    ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    results_volume.commit()
    return "completed"


@app.local_entrypoint()
def main():
    print(run_pipeline.remote())
