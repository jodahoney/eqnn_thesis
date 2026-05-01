from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from eqnn.datasets.heisenberg import DatasetSplit, HeisenbergDatasetConfig, generate_dataset
from eqnn.models.qcnn import QCNNConfig, SU2QCNN
from eqnn.training.loop import Trainer, TrainingConfig


class TinySymmetryModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_qubits=1)
        self.parameters = np.asarray([0.0], dtype=np.float64)

    def get_parameters(self) -> np.ndarray:
        return self.parameters.copy()

    def set_parameters(self, parameters: np.ndarray) -> None:
        self.parameters = np.asarray(parameters, dtype=np.float64).copy()

    def predict(self, state: np.ndarray, parameters: np.ndarray | None = None) -> float:
        parameter_array = self.parameters if parameters is None else np.asarray(parameters, dtype=np.float64)
        return float(np.clip(np.real(state[0]) + parameter_array[0], 0.0, 1.0))

    def predict_batch(self, states: np.ndarray, parameters: np.ndarray | None = None) -> np.ndarray:
        return np.asarray([self.predict(state, parameters=parameters) for state in states], dtype=np.float64)


class TrainerTests(unittest.TestCase):
    def _combined_small_dataset(self) -> DatasetSplit:
        bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=4,
                ratio_min=0.4,
                ratio_max=1.6,
                num_points=9,
                split_seed=3,
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

    def test_training_reduces_loss_on_small_heisenberg_dataset(self) -> None:
        dataset = self._combined_small_dataset()
        model = SU2QCNN(
            QCNNConfig(
                num_qubits=4,
                min_readout_qubits=4,
                readout_mode="dimerization",
            )
        )
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

    def test_swap_readout_default_initialization_has_zero_gradient(self) -> None:
        dataset = self._combined_small_dataset()
        model = SU2QCNN(QCNNConfig(num_qubits=4))
        trainer = Trainer(TrainingConfig())

        gradient = trainer.gradient(model, dataset)

        np.testing.assert_allclose(gradient, np.zeros_like(gradient), atol=1e-12)

    def test_swap_readout_training_recovers_with_noisy_initialization(self) -> None:
        dataset = self._combined_small_dataset()
        model = SU2QCNN(QCNNConfig(num_qubits=4))
        trainer = Trainer(
            TrainingConfig(
                epochs=40,
                learning_rate=0.1,
                finite_difference_eps=1e-3,
                gradient_backend="exact",
                initialization_strategy="noisy_current",
                initialization_noise_scale=0.05,
                random_seed=0,
            )
        )

        history = trainer.fit(model, dataset)

        self.assertLess(history["best_loss"], history["loss"][0] - 0.4)
        self.assertGreaterEqual(history["best_accuracy"], 0.95)
        self.assertEqual(history["best_restart"], 0)

    def test_paper_threshold_update_uses_nearest_points_across_the_transition(self) -> None:
        class DummyThresholdModel:
            def __init__(self) -> None:
                self.threshold = 0.5

            def set_classification_threshold(self, threshold: float) -> None:
                self.threshold = float(threshold)

            def get_classification_threshold(self) -> float:
                return self.threshold

            def predict_batch(self, states: np.ndarray, parameters: np.ndarray | None = None) -> np.ndarray:
                return np.asarray(np.real(states[:, 0]), dtype=np.float64)

        split = DatasetSplit(
            states=np.asarray([[0.1], [0.2], [0.6], [0.9]], dtype=np.complex128),
            labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            coupling_ratios=np.asarray([0.2, 0.9, 1.1, 1.8], dtype=np.float64),
            ground_state_energies=np.zeros(4, dtype=np.float64),
        )
        trainer = Trainer(
            TrainingConfig(
                loss="mse",
                batch_size=2,
                threshold_update="paper_nearest_critical",
                threshold_critical_ratio=1.0,
            )
        )
        model = DummyThresholdModel()

        trainer._maybe_update_classification_threshold(model, split, np.zeros(0, dtype=np.float64))

        self.assertAlmostEqual(model.get_classification_threshold(), 0.4, places=12)

    def test_training_config_validates_symmetry_regularization_options(self) -> None:
        config = TrainingConfig()

        self.assertFalse(config.symmetry_regularization)
        self.assertEqual(config.symmetry_regularization_weight, 0.0)

        with self.assertRaisesRegex(ValueError, "symmetry_regularization_weight"):
            TrainingConfig(symmetry_regularization=True, symmetry_regularization_weight=-0.1)

        with self.assertRaisesRegex(ValueError, "num_symmetry_regularization_samples"):
            TrainingConfig(symmetry_regularization=True, num_symmetry_regularization_samples=0)

        with self.assertRaisesRegex(ValueError, "symmetry_regularization_frequency"):
            TrainingConfig(symmetry_regularization=True, symmetry_regularization_frequency=0)

        with self.assertRaisesRegex(ValueError, "symmetry_regularization_state_samples"):
            TrainingConfig(symmetry_regularization=True, symmetry_regularization_state_samples=0)

    def test_symmetry_regularized_training_records_penalty_history(self) -> None:
        split = DatasetSplit(
            states=np.asarray(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j],
                ],
                dtype=np.complex128,
            ),
            labels=np.asarray([1, 0], dtype=np.int64),
            coupling_ratios=np.asarray([0.5, 1.5], dtype=np.float64),
            ground_state_energies=np.zeros(2, dtype=np.float64),
        )
        trainer = Trainer(
            TrainingConfig(
                epochs=1,
                learning_rate=0.01,
                loss="mse",
                gradient_backend="finite_difference",
                symmetry_regularization=True,
                symmetry_regularization_weight=0.1,
                num_symmetry_regularization_samples=1,
                symmetry_regularization_state_samples=1,
                symmetry_regularization_seed=0,
            )
        )

        history = trainer.fit(TinySymmetryModel(), split)

        self.assertIn("symmetry_penalty", history)
        self.assertIn("weighted_symmetry_penalty", history)
        self.assertEqual(history["symmetry_regularization_note"], "finite_difference_objective_regularizer")
        self.assertEqual(len(history["symmetry_penalty"]), 2)
        self.assertTrue(all(value >= 0.0 for value in history["symmetry_penalty"]))
