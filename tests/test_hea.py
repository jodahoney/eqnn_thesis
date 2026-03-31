from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eqnn.backends import NumpyPureStateBackend
from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.experiments import ExperimentConfig, run_training_experiment
from eqnn.layers import HEAConvolution, HEAConvolutionConfig
from eqnn.models import HEAQCNN, HEAQCNNConfig
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.training import TrainingConfig
from eqnn.verification.equivariance import model_invariance_error


class HEALayerTests(unittest.TestCase):
    def test_hea_block_gate_is_unitary(self) -> None:
        gate = HEAConvolution.gate(0.1, -0.2, 0.05, 0.17, -0.11, 0.08, 0.07, -0.03)
        identity = np.eye(4, dtype=np.complex128)
        np.testing.assert_allclose(gate.conjugate().T @ gate, identity, atol=1e-12)

    def test_hea_convolution_parameter_count_tracks_active_parities(self) -> None:
        layer = HEAConvolution(HEAConvolutionConfig(num_qubits=4))
        self.assertEqual(layer.parameter_count, 16)

    def test_hea_unitary_and_gradients_match_finite_difference(self) -> None:
        parameters = np.asarray((0.1, -0.2, 0.05, 0.17, -0.11, 0.08, 0.07, -0.03), dtype=np.float64)
        layer = HEAConvolution(
            HEAConvolutionConfig(num_qubits=2, parity_sequence=("even",)),
            parameters=parameters,
        )

        _, gradients = layer.unitary_and_gradients()
        self.assertEqual(len(gradients), 8)

        eps = 1e-6
        for index, analytic_gradient in enumerate(gradients):
            offset = np.zeros_like(parameters)
            offset[index] = eps
            plus = layer.unitary(parameters=parameters + offset)
            minus = layer.unitary(parameters=parameters - offset)
            finite_difference = (plus - minus) / (2.0 * eps)
            np.testing.assert_allclose(analytic_gradient, finite_difference, atol=1e-6)


class HEAQCNNTests(unittest.TestCase):
    def test_hea_qcnn_forward_output_is_probabilistic(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.6)
        model = HEAQCNN(
            HEAQCNNConfig(num_qubits=4),
            parameters=np.linspace(-0.2, 0.2, 24),
        )

        forward = model.forward(state)

        self.assertGreaterEqual(forward.probability, 0.0)
        self.assertLessEqual(forward.probability, 1.0)
        self.assertEqual(forward.final_num_qubits, 2)
        self.assertEqual(forward.readout_mode, "swap")

    def test_hea_qcnn_predict_batch_returns_probabilities(self) -> None:
        dataset = generate_dataset(HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=2))
        states = np.concatenate([dataset.train.states, dataset.test.states], axis=0)
        model = HEAQCNN(HEAQCNNConfig(num_qubits=4))

        probabilities = model.predict_batch(states)

        self.assertEqual(probabilities.shape, (states.shape[0],))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_explicit_numpy_backend_matches_default_hea_outputs(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.6)
        parameters = np.linspace(-0.15, 0.25, 24)

        default_model = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=parameters)
        explicit_backend_model = HEAQCNN(
            HEAQCNNConfig(num_qubits=4),
            parameters=parameters,
            backend=NumpyPureStateBackend(),
        )

        default_forward = default_model.forward(state)
        explicit_forward = explicit_backend_model.forward(state)

        np.testing.assert_allclose(
            explicit_forward.final_density_matrix,
            default_forward.final_density_matrix,
            atol=1e-12,
        )
        self.assertAlmostEqual(explicit_forward.probability, default_forward.probability, places=12)

    def test_hea_qcnn_breaks_su2_invariance(self) -> None:
        model = HEAQCNN(
            HEAQCNNConfig(num_qubits=4),
            parameters=np.linspace(-0.2, 0.2, 24),
        )

        summary = model_invariance_error(model, num_trials=5, seed=11)
        self.assertGreater(summary["max_error"], 1e-3)

    def test_hea_qcnn_runs_through_experiment_runner(self) -> None:
        dataset = generate_dataset(HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=3))
        training_config = TrainingConfig(epochs=1, learning_rate=0.05, gradient_backend="exact")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "hea_experiment"
            result = run_training_experiment(
                dataset,
                ExperimentConfig(model_family="hea_qcnn", num_qubits=4),
                training_config,
                output_dir=output_dir,
            )

            self.assertEqual(result["experiment_config"]["model_family"], "hea_qcnn")
            self.assertTrue((output_dir / "metrics.json").exists())
