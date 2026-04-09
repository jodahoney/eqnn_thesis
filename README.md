# EQNN Simulator

This repository is now a working research scaffold for small-scale equivariant quantum neural network experiments around the bond-alternating Heisenberg chain. It supports:

- paper-style SU(2)-EQCNN models with `exp(-i theta SWAP)` brickwork convolutions,
- QCNN-shaped non-equivariant comparators (`hea_qcnn` and the older anisotropic baseline),
- multiple execution backends,
- paper-style dataset generation and reproduction workflows,
- calibration sweeps for practical training budgets,
- and first-pass noisy mixed-state comparisons through the Qiskit mixed backend.

The codebase is organized around a `trainer -> model -> backend` split, so model semantics stay stable while numerical execution can vary by backend.

## Current Status

### Implemented model families

- `su2_qcnn`
  Paper-style SU(2)-equivariant QCNN with shared brickwork `exp(-i theta SWAP)` convolutions.
- `hea_qcnn`
  QCNN hierarchy with HEA-style two-qubit convolution blocks for the paper-aligned non-equivariant comparator.
- `baseline_qcnn`
  Older anisotropic QCNN baseline retained for ablations and comparison.

### Implemented backends

- `numpy_pure`
  Reference backend for the current noiseless simulator path.
- `torch_pure`
  Torch-native pure-state backend with batched prediction/evaluation support.
- `qiskit_pure`
  Correctness-first Qiskit parity backend for small noiseless runs.
- `qiskit_mixed`
  Mixed-state Qiskit backend for small density-matrix experiments with simple explicit noise channels.

### Implemented experiment workflows

- dataset generation for the bond-alternating Heisenberg model
- paper-style locked reproduction runs
- generic experiment runs
- backend comparison benchmarks
- calibration sweeps for choosing a practical epoch budget
- small noisy mixed-state comparison sweeps

## Current Scientific Assumptions

The default problem is the spin-1/2 bond-alternating Heisenberg chain

\[
H(r) = \sum_{b \in \text{primary bonds}} \vec{S}_i \cdot \vec{S}_j
+ r \sum_{b \in \text{secondary bonds}} \vec{S}_i \cdot \vec{S}_j,
\]

with open boundaries by default and alternating couplings along the chain.

The dataset layer supports:

- ground-state generation by exact diagonalization,
- stored diagnostics including singlet fractions, dimerization, and partial reflection,
- conservative ratio-threshold labels,
- and paper-style train/test grids for reproduction and comparison workflows.

For the locked paper-style path, labels are aligned with the SWAP readout convention: larger SWAP outputs correspond to the trivial side of the transition.

## Project Layout

```text
eqnn/
  backends/      Numerical execution backends (NumPy, Torch, Qiskit)
  circuits/      Qiskit circuit-construction helpers
  datasets/      Dataset generation, caching, and serialization
  experiments/   Experiment runners, calibration, reproduction, noisy comparisons
  groups/        Symmetry-group representations and rotations
  layers/        Convolution and pooling layers
  models/        QCNN model definitions and trainable interfaces
  noise/         Mixed-state noise configuration helpers
  physics/       Hamiltonians, observables, and linear-algebra utilities
  training/      Optimization and evaluation loop
  verification/  Symmetry and sanity checks
tests/           Regression and smoke tests
```

## Installation

Base install:

```bash
python3 -m pip install -e .
```

Optional extras:

```bash
python3 -m pip install -e ".[torch]"
python3 -m pip install -e ".[qiskit]"
python3 -m pip install -e ".[torch,qiskit]"
```

If optional dependencies are missing, the corresponding Torch/Qiskit tests skip cleanly and the backends raise a clear import error when instantiated.

## Core Workflows

### 1. Generate a dataset

```bash
python3 -m eqnn generate-dataset \
  --num-qubits 6 \
  --num-points 31 \
  --ratio-min 0.4 \
  --ratio-max 1.6 \
  --eigensolver auto \
  --labeling-strategy partial_reflection \
  --diagnostic-window 0.02 \
  --output-dir data/generated/heisenberg_n6
```

This writes `train.npz`, `test.npz`, and `metadata.json`.

### 2. Run a single experiment

```bash
python3 -m eqnn run-experiment \
  --num-qubits 4 \
  --num-points 9 \
  --model-family hea_qcnn \
  --backend numpy_pure \
  --gradient-backend exact \
  --epochs 20 \
  --learning-rate 0.05 \
  --output-dir data/experiments/hea_n4
```

Artifacts include `metrics.json`, `best_parameters.npy`, `train_predictions.npz`, and `test_predictions.npz`.

### 3. Compare backends on the same experiment

```bash
python3 -m eqnn benchmark-backends \
  --backends numpy_pure torch_pure \
  --num-qubits 6 \
  --model-family su2_qcnn \
  --epochs 10 \
  --learning-rate 0.05 \
  --output-dir data/backend_benchmarks/n6_su2
```

### 4. Run the locked paper reproduction baseline

```bash
python3 -m eqnn run-paper-reproduction \
  --num-qubits 6 \
  --train-sizes 2 4 6 8 10 12 \
  --random-seeds 0 1 2 \
  --epochs 750 \
  --output-dir data/reproduction/paper_reproduction_v1_n6
```

This path is locked to the paper-style SU(2)-EQCNN setting:

- `model_family=su2_qcnn`
- `pooling_mode=partial_trace`
- `readout_mode=swap`
- shared convolution parameters
- open-boundary Heisenberg data
- `ADAM`, `MSE`, batch size `2`
- paper-style threshold updates

### 5. Run a calibration sweep

```bash
python3 -m eqnn run-calibration-sweep \
  --model-families su2_qcnn hea_qcnn \
  --num-qubits-values 6 10 \
  --train-sizes 12 \
  --epochs-values 50 150 300 500 750 \
  --random-seeds 0 1 2 \
  --output-dir data/calibration/epoch_budget_run
```

This is the compact workflow for selecting a practical global epoch budget before larger comparison studies.

### 6. Run a small noisy mixed-state comparison

```bash
python3 -m eqnn run-noisy-comparison \
  --model-families su2_qcnn hea_qcnn \
  --num-qubits-values 4 6 \
  --train-sizes 4 8 \
  --epochs-values 10 \
  --random-seeds 0 1 \
  --backend-name qiskit_mixed \
  --noise-model-name depolarizing \
  --noise-strength-values 0.0 0.001 0.005 0.01 \
  --output-dir data/noisy/depolarizing_small
```

This is the first small-scale noisy workflow intended for scientifically meaningful mixed-state results at tractable qubit counts.

## Training Notes

- The default `swap` readout is the paper-style terminal readout:

\[
f_\theta(\rho) = \frac{\mathrm{Tr}[\phi_\theta(\rho)\,\mathrm{SWAP}] + 1}{2}.
\]

- `readout_mode="dimerization"` is still available for legacy comparisons and optimization baselines.
- Noisy warm starts and restarts are supported through `TrainingConfig(initialization_strategy="noisy_current", initialization_noise_scale=..., num_restarts=...)`.
- `gradient_backend="auto"` uses exact gradients when the backend supports them and otherwise falls back to finite differences.

### Current gradient policy by backend

- `numpy_pure`
  Exact-gradient path supported for the current pure-state QCNN route.
- `torch_pure`
  Exact gradients supported for the current partial-trace pure-state path.
- `qiskit_pure`
  Finite-difference training only.
- `qiskit_mixed`
  Finite-difference training only.

## Current Limitations

- The Qiskit backends are correctness-first and intended for small qubit counts.
- `qiskit_mixed` currently supports simple explicit noise channels, not realistic device calibration models.
- Mixed-state noisy workflows are meant for low-`n` studies first, not large sweeps.
- The paper-style reproduction path is the most faithful locked baseline; the broader experiment workflows are more flexible and may include simplifications depending on configuration.

## Sherlock / Cluster Usage

For the paper reproduction workflow on Stanford Sherlock, the repo includes
[sherlock_paper_reproduction.sbatch](/Users/joda/Desktop/stanford/eqnn_thesis/scripts/sherlock_paper_reproduction.sbatch).

With the repo checked out at `/scratch/users/jdehoney/eqnn_thesis`:

```bash
mkdir -p /scratch/users/jdehoney/eqnn_thesis/logs/slurm
cd /scratch/users/jdehoney/eqnn_thesis
sbatch scripts/sherlock_paper_reproduction.sbatch
```

Useful overrides:

```bash
cd /scratch/users/jdehoney/eqnn_thesis
MODULE_PYTHON=python/3.11.9 VENV_DIR="$HOME/venvs/eqnn" \
sbatch --array=6-8 scripts/sherlock_paper_reproduction.sbatch
```

```bash
cd /scratch/users/jdehoney/eqnn_thesis
RUN_ID=paper_reproduction_smoke OUTPUT_ROOT="$PWD/data/reproduction/sherlock_test" \
sbatch --array=6 scripts/sherlock_paper_reproduction.sbatch
```

## Tests

Run the full suite with:

```bash
python3 -m unittest discover -s tests
```

Optional-dependency tests for Torch and Qiskit will skip unless those packages are installed.
