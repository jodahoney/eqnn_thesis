from __future__ import annotations

import unittest

import numpy as np

from eqnn.backends import NumpyPureStateBackend
from eqnn.datasets.heisenberg import DatasetSplit, HeisenbergDatasetConfig, generate_dataset
from eqnn.models import BaselineQCNN, BaselineQCNNConfig, QCNNConfig, SU2QCNN
from eqnn.training import Trainer, TrainingConfig


class ExactGradientTests(unittest.TestCase):
    def _combined_dataset(self, *, num_qubits: int, num_points: int, split_seed: int) -> DatasetSplit:
        bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=num_qubits,
                num_points=num_points,
                split_seed=split_seed,
            )
        )
        return DatasetSplit(
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

    def test_su2_swap_exact_gradient_matches_finite_difference(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=1)
        parameters = np.asarray((0.12, -0.08, 0.17), dtype=np.float64)
        model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters)

        exact_trainer = Trainer(TrainingConfig(gradient_backend="exact"))
        finite_difference_trainer = Trainer(TrainingConfig(gradient_backend="finite_difference"))

        exact_gradient = exact_trainer.gradient(model, dataset)
        finite_difference_gradient = finite_difference_trainer.gradient(model, dataset)

        np.testing.assert_allclose(exact_gradient, finite_difference_gradient, atol=1e-5)

    def test_baseline_dimerization_exact_gradient_matches_finite_difference(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=2)
        parameters = np.asarray((0.15, -0.11, 0.07, 0.05, 0.09, -0.13, 0.8, -0.2), dtype=np.float64)
        model = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4, min_readout_qubits=4, readout_mode="dimerization"),
            parameters=parameters,
        )

        exact_trainer = Trainer(TrainingConfig(gradient_backend="exact"))
        finite_difference_trainer = Trainer(TrainingConfig(gradient_backend="finite_difference"))

        exact_gradient = exact_trainer.gradient(model, dataset)
        finite_difference_gradient = finite_difference_trainer.gradient(model, dataset)

        np.testing.assert_allclose(exact_gradient, finite_difference_gradient, atol=1e-5)

    def test_equivariant_pooling_exact_gradient_matches_finite_difference(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=3)
        parameters = np.asarray((0.11, -0.04, 0.2, -0.1, 0.3, -0.2), dtype=np.float64)
        model = SU2QCNN(
            QCNNConfig(num_qubits=4, pooling_mode="equivariant"),
            parameters=parameters,
        )

        exact_trainer = Trainer(TrainingConfig(gradient_backend="exact"))
        finite_difference_trainer = Trainer(TrainingConfig(gradient_backend="finite_difference"))

        exact_gradient = exact_trainer.gradient(model, dataset)
        finite_difference_gradient = finite_difference_trainer.gradient(model, dataset)

        np.testing.assert_allclose(exact_gradient, finite_difference_gradient, atol=1e-5)

    def test_su2_swap_mse_exact_gradient_matches_finite_difference(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=4)
        parameters = np.asarray((0.09, -0.03, 0.14), dtype=np.float64)
        model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters)

        exact_trainer = Trainer(TrainingConfig(loss="mse", gradient_backend="exact"))
        finite_difference_trainer = Trainer(
            TrainingConfig(loss="mse", gradient_backend="finite_difference")
        )

        exact_gradient = exact_trainer.gradient(model, dataset)
        finite_difference_gradient = finite_difference_trainer.gradient(model, dataset)

        np.testing.assert_allclose(exact_gradient, finite_difference_gradient, atol=1e-5)

    def test_trainer_falls_back_when_backend_lacks_exact_gradients(self) -> None:
        class NoExactGradientBackend(NumpyPureStateBackend):
            @property
            def supports_exact_gradients(self) -> bool:
                return False

            def loss_gradient(self, *args, **kwargs) -> np.ndarray:
                raise NotImplementedError

        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=5)
        parameters = np.asarray((0.12, -0.08, 0.17), dtype=np.float64)
        model = SU2QCNN(
            QCNNConfig(num_qubits=4),
            parameters=parameters,
            backend=NoExactGradientBackend(),
        )

        auto_trainer = Trainer(TrainingConfig(gradient_backend="auto"))
        finite_difference_trainer = Trainer(TrainingConfig(gradient_backend="finite_difference"))

        auto_gradient = auto_trainer.gradient(model, dataset)
        finite_difference_gradient = finite_difference_trainer.gradient(model, dataset)

        np.testing.assert_allclose(auto_gradient, finite_difference_gradient, atol=1e-5)
