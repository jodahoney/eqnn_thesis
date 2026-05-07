#!/bin/bash
#SBATCH --job-name=all_noise_n7_arr
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-239
#SBATCH --output=logs/all_noise_n7_array_%A_%a.out
#SBATCH --error=logs/all_noise_n7_array_%A_%a.err

set -euo pipefail

cd /scratch/users/jdehoney/eqnn_thesis
mkdir -p logs

module load python/3.12.1
module load openblas/0.3.28

source /scratch/users/jdehoney/venvs/eqnn_py312/bin/activate

export OUTPUT_ROOT="data/noisy_comparisons/all_noise_mitigation_n7_array"

python3 - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
output_root = os.environ["OUTPUT_ROOT"]

model_families = ["su2_qcnn", "hea_qcnn"]
train_sizes = [16, 32]
seeds = [0, 1, 2]
noise_models = [
    "depolarizing",
    "phase_damping",
    "amplitude_damping",
    "coherent_overrotation",
]

eval_noise_strengths = ["0.0", "0.01", "0.03", "0.05", "0.075", "0.1"]
train_noise_strengths = ["0.01", "0.03", "0.05"]

# Same mitigation grid for both SU2-QCNN and HEA-QCNN.
# This makes the comparison conceptually symmetric:
#   none
#   noise-aware training
#   symmetry regularization beta sweep
mitigation_configs = [
    {
        "method": "none",
        "beta_values": None,
        "train_noise_values": None,
    },
    {
        "method": "noise_aware_training",
        "beta_values": None,
        "train_noise_values": train_noise_strengths,
    },
    {
        "method": "symmetry_regularized",
        "beta_values": ["0.01"],
        "train_noise_values": None,
    },
    {
        "method": "symmetry_regularized",
        "beta_values": ["0.05"],
        "train_noise_values": None,
    },
    {
        "method": "symmetry_regularized",
        "beta_values": ["0.1"],
        "train_noise_values": None,
    },
]

grid = []

for model_family in model_families:
    for train_size in train_sizes:
        for seed in seeds:
            for noise_model in noise_models:
                for mitigation_config in mitigation_configs:
                    grid.append(
                        {
                            "model_family": model_family,
                            "train_size": train_size,
                            "seed": seed,
                            "noise_model": noise_model,
                            **mitigation_config,
                        }
                    )

expected_grid_size = 240
if len(grid) != expected_grid_size:
    raise SystemExit(
        f"Internal grid-size mismatch: expected {expected_grid_size}, got {len(grid)}"
    )

if task_id >= len(grid):
    raise SystemExit(f"Task id {task_id} outside grid size {len(grid)}")

cfg = grid[task_id]

print("===== Array task config =====")
print(f"task_id: {task_id}")
print(f"grid_size: {len(grid)}")
for key in [
    "model_family",
    "train_size",
    "seed",
    "noise_model",
    "method",
    "beta_values",
    "train_noise_values",
]:
    print(f"{key}: {cfg[key]}")
print(f"output_root: {output_root}")
print()

cmd = [
    sys.executable,
    "-m",
    "eqnn",
    "run-all-noise-mitigation-comparison",
    "--backend-name",
    "qiskit_mixed",
    "--model-families",
    cfg["model_family"],
    "--num-qubits-values",
    "7",
    "--train-sizes",
    str(cfg["train_size"]),
    "--epochs-values",
    "20",
    "--random-seeds",
    str(cfg["seed"]),
    "--noise-model-names",
    cfg["noise_model"],
    "--noise-strength-values",
    *eval_noise_strengths,
    "--mitigation-methods",
    cfg["method"],
    "--output-root",
    output_root,
]

if cfg["train_noise_values"] is not None:
    cmd += ["--train-noise-strength-values", *cfg["train_noise_values"]]

if cfg["beta_values"] is not None:
    cmd += ["--symmetry-regularization-beta-values", *cfg["beta_values"]]

print("===== Command =====")
print(" ".join(cmd))
print()

Path(output_root).mkdir(parents=True, exist_ok=True)
subprocess.run(cmd, check=True)

print()
print("===== Task completed successfully =====")
PY
