# EQNN Simulator

This repository now contains a working first pass of an equivariant quantum neural network simulator:

- a bond-alternating Heisenberg dataset generator for small spin chains,
- exact diagonalization of the Hamiltonian to obtain ground states,
- phase labels derived from the coupling ratio,
- an SU(2)-equivariant convolution layer built from `exp(-i theta SWAP)`,
- a physically valid pooling layer based on partial trace,
- a QCNN forward pass with an SU(2)-invariant dimerization readout,
- a small training loop and symmetry-verification utilities.

## Current assumptions

The Stage 1 generator uses the spin-1/2 bond-alternating Heisenberg Hamiltonian

\[
H(r) = \sum_{b \in \text{primary bonds}} \vec{S}_i \cdot \vec{S}_j
+ r \sum_{b \in \text{secondary bonds}} \vec{S}_i \cdot \vec{S}_j,
\]

with the bond `(0, 1)` assigned coupling `1.0`, the bond `(1, 2)` assigned coupling `r`, and the pattern alternating along the chain.

Labels are assigned with a simple phase proxy:

- `label = 0` when `r < critical_ratio - exclusion_window`
- `label = 1` when `r > critical_ratio + exclusion_window`

By default the critical ratio is `1.0`, and samples too close to the transition are excluded from the dataset.

## Project layout

```text
eqnn/
  datasets/      Dataset generation and serialization
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
  --output-dir data/generated/heisenberg_n6
```

This writes `train.npz`, `test.npz`, and `metadata.json` inside the requested output directory.

## Train a small QCNN

The Python API now supports an end-to-end minimal experiment:

```python
from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.models.qcnn import QCNNConfig, SU2QCNN
from eqnn.training.loop import Trainer, TrainingConfig

dataset = generate_dataset(HeisenbergDatasetConfig(num_qubits=4))
model = SU2QCNN(QCNNConfig(num_qubits=4, min_readout_qubits=4))
trainer = Trainer(TrainingConfig(epochs=20, learning_rate=0.1))
history = trainer.fit(model, dataset.train)
```

## Run the tests

```bash
python3 -m unittest discover -s tests
```
