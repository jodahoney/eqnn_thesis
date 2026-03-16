"""EQNN simulator package."""

from eqnn.datasets.heisenberg import (
    DatasetBundle,
    DatasetSplit,
    HeisenbergDatasetConfig,
    generate_dataset,
)
from eqnn.models.qcnn import QCNNConfig, QCNNForwardPass, SU2QCNN
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.training.loop import Trainer, TrainingConfig

__all__ = [
    "BondAlternatingHeisenbergHamiltonian",
    "DatasetBundle",
    "DatasetSplit",
    "HeisenbergDatasetConfig",
    "QCNNConfig",
    "QCNNForwardPass",
    "SU2QCNN",
    "Trainer",
    "TrainingConfig",
    "generate_dataset",
]
