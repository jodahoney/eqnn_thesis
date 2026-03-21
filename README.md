# EQNN Simulator

This repository now contains a working first pass of an equivariant quantum neural network simulator:

- a bond-alternating Heisenberg dataset generator for small spin chains,
- exact diagonalization of the Hamiltonian to obtain ground states,
- phase diagnostics including dimerization and partial reflection, with configurable labeling strategies,
- an SU(2)-equivariant convolution layer built from `exp(-i theta SWAP)`,
- a physically valid pooling layer based on partial trace,
- a trainable SU(2)-equivariant pooling family using the full paper-style `2 -> 1` CPTP channel basis,
- a QCNN forward pass with the paper-style two-qubit `SWAP` readout,
- a legacy SU(2)-invariant dimerization readout for comparison and training baselines,
- a symmetry-agnostic anisotropic QCNN baseline,
- experiment and benchmark-sweep utilities that save metrics and predictions,
- a small training loop and symmetry-verification utilities.

## Current assumptions

The Stage 1 generator uses the spin-1/2 bond-alternating Heisenberg Hamiltonian

\[
H(r) = \sum_{b \in \text{primary bonds}} \vec{S}_i \cdot \vec{S}_j
+ r \sum_{b \in \text{secondary bonds}} \vec{S}_i \cdot \vec{S}_j,
\]

with the bond `(0, 1)` assigned coupling `1.0`, the bond `(1, 2)` assigned coupling `r`, and the pattern alternating along the chain.

The dataset generator now stores several phase diagnostics for every sample:

- primary and secondary singlet fractions,
- the dimerization feature `secondary - primary`,
- the normalized partial-reflection invariant on a central reflected region.

By default labels still use the conservative ratio-threshold rule

- `label = 0` when `r < critical_ratio - exclusion_window`
- `label = 1` when `r > critical_ratio + exclusion_window`

but the CLI and Python config also support `labeling_strategy="partial_reflection"` for even-length chains, which uses the finite-size many-body diagnostic instead.

## Project layout

```text
eqnn/
  datasets/      Dataset generation and serialization
  experiments/   Experiment runners and benchmark sweeps
  groups/        Symmetry-group representations and rotations
  physics/       Spin operators and Hamiltonians
  layers/        SU(2) convolution and pooling layers
  models/        QCNN forward pass and readout
  training/      Small-sample optimization utilities
  verification/  Numerical symmetry checks
tests/           Dataset, layer, model, and training tests
```

## Install dependencies

```bash
python3 -m pip install -e .
```

## Generate a dataset

You can run the dataset generator directly from the repository root:

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

This writes `train.npz`, `test.npz`, and `metadata.json` inside the requested output directory. The split files now include saved diagnostic arrays in addition to states, labels, ratios, and energies.

For larger chains, `--eigensolver sparse` switches ground-state solves to a sparse SciPy backend. The default `auto` mode keeps dense solves for small systems and switches to sparse solves when the Hilbert space is large enough to benefit.

## Train a small QCNN

The Python API now supports an end-to-end minimal experiment:

```python
from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.models.qcnn import QCNNConfig, SU2QCNN
from eqnn.training.loop import Trainer, TrainingConfig

dataset = generate_dataset(HeisenbergDatasetConfig(num_qubits=4))
model = SU2QCNN(
    QCNNConfig(
        num_qubits=4,
        min_readout_qubits=4,
        readout_mode="dimerization",
    )
)
trainer = Trainer(TrainingConfig(epochs=20, learning_rate=0.1))
history = trainer.fit(model, dataset.train)
```

The default `QCNNConfig(num_qubits=...)` now uses the paper-style final readout

\[
f_\theta(\rho) = \frac{\mathrm{Tr}[\phi_\theta(\rho)\,\mathrm{SWAP}] + 1}{2},
\]

so the model automatically pools down to two qubits before evaluation. The older dimerization-based readout is still available via `readout_mode="dimerization"` and remains a useful optimization baseline for ablations and baseline comparisons.

For the swap-readout path, the all-zero convolution initialization sits on a symmetric zero-gradient point for the small 4-qubit benchmark. The trainer now supports noisy warm starts and restarts through `TrainingConfig(initialization_strategy="noisy_current", initialization_noise_scale=..., num_restarts=..., random_seed=...)`, which makes the paper-style readout trainable without changing the model definition.

The trainer also now supports `gradient_backend="auto" | "exact" | "finite_difference"`. The default `auto` mode uses the exact adjoint-style gradient path across the QCNN convolution, pooling, and readout parameters, while `finite_difference` remains available as a verification fallback.

If you want the paper-derived pooling family instead of fixed partial trace, set `pooling_mode="equivariant"` in `QCNNConfig`.

Training now also supports explicit `loss="bce" | "mse"`, minibatching via `batch_size`, and paper-style threshold updates via `threshold_update="paper_nearest_critical"`.

## Run an experiment

You can train either the equivariant QCNN or the symmetry-agnostic baseline and save artifacts:

```bash
python3 -m eqnn run-experiment \
  --num-qubits 4 \
  --num-points 9 \
  --model-family baseline_qcnn \
  --readout-mode dimerization \
  --min-readout-qubits 4 \
  --gradient-backend exact \
  --epochs 20 \
  --learning-rate 0.1 \
  --output-dir data/experiments/baseline_n4
```

This writes `metrics.json`, `best_parameters.npy`, `train_predictions.npz`, and `test_predictions.npz` to the requested output directory.

## Run a benchmark sweep

For a small comparison grid:

```bash
python3 -m eqnn run-benchmark-sweep \
  --num-qubits-values 4 6 \
  --model-families su2_qcnn baseline_qcnn \
  --labeling-strategies ratio_threshold \
  --pooling-modes partial_trace \
  --readout-modes dimerization \
  --epochs 10 \
  --learning-rate 0.05 \
  --output-dir data/benchmarks/small_grid
```

The sweep writes per-experiment artifact directories, cached generated datasets, plus `summary.json` and `summary.csv`.

If a longer sweep is still running or was interrupted, you can aggregate whatever has already completed:

```bash
python3 -m eqnn summarize-experiments \
  --input-dir data/benchmarks/small_grid/experiments \
  --num-qubits 4 \
  --output-json data/benchmarks/small_grid/n4_summary.json \
  --output-csv data/benchmarks/small_grid/n4_summary.csv
```

## Run the Paper Reproduction Baseline

The repo now includes a locked `paper_reproduction_v1` path that fixes:

- `model_family=su2_qcnn`
- `pooling_mode=partial_trace`
- `readout_mode=swap`
- shared SU(2)-equivariant brickwork convolutions
- open-boundary bond-alternating Heisenberg data
- `ADAM`, `MSE`, batch size `2`
- threshold initialization at `0.5` with epochwise updates from the nearest points to the critical ratio

For a smoke run:

```bash
python3 -m eqnn run-paper-reproduction \
  --num-qubits 4 \
  --train-sizes 2 4 6 \
  --random-seeds 0 1 2 \
  --epochs 30 \
  --output-dir data/reproduction/paper_reproduction_v1
```

This writes per-seed experiment artifacts, per-train-size phase-diagram summaries, and aggregate `summary.json` / `summary.csv`.

## Run the tests

```bash
python3 -m unittest discover -s tests
```
