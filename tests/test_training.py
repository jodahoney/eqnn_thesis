from __future__ import annotations

import unittest

import numpy as np

from eqnn.datasets.heisenberg import DatasetSplit, HeisenbergDatasetConfig, generate_dataset
from eqnn.models.qcnn import QCNNConfig, SU2QCNN
from eqnn.training.loop import Trainer, TrainingConfig


class TrainerTests(unittest.TestCase):
    def test_training_reduces_loss_on_small_heisenberg_dataset(self) -> None:
        bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=4,
                ratio_min=0.4,
                ratio_max=1.6,
                num_points=9,
                split_seed=3,
            )
        )
        dataset = DatasetSplit(
            states=np.concatenate([bundle.train.states, bundle.test.states], axis=0),
            labels=np.concatenate([bundle.train.labels, bundle.test.labels], axis=0),
            coupling_ratios=np.concatenate(
                [bundle.train.coupling_ratios, bundle.test.coupling_ratios],
                axis=0,
            ),
            ground_state_energies=np.concatenate(
                [bundle.train.ground_state_energies, bundle.test.ground_state_energies],
                axis=0,
            ),
        )
        model = SU2QCNN(QCNNConfig(num_qubits=4, min_readout_qubits=4))
        trainer = Trainer(
            TrainingConfig(
                epochs=20,
                learning_rate=0.1,
                finite_difference_eps=1e-3,
            )
        )

        history = trainer.fit(model, dataset)

        self.assertLess(history["best_loss"], history["loss"][0] - 0.05)
        self.assertGreaterEqual(history["best_accuracy"], 0.75)
