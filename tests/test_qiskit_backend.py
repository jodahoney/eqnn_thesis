from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eqnn.backends import NumpyPureStateBackend, QISKIT_AVAILABLE, QiskitMixedStateBackend, QiskitPureStateBackend
from eqnn.datasets.heisenberg import DatasetSplit, HeisenbergDatasetConfig, generate_dataset
from eqnn.experiments.runner import ExperimentConfig, build_backend, run_training_experiment
from eqnn.models import HEAQCNN, HEAQCNNConfig, QCNNConfig, SU2QCNN
from eqnn.noise import NoiseConfig
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.training import TrainingConfig


@unittest.skipUnless(QISKIT_AVAILABLE, "qiskit is not installed")
class QiskitBackendTests(unittest.TestCase):
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
            coupling_ratios=np.concatenate([bundle.train.coupling_ratios, bundle.test.coupling_ratios], axis=0),
            ground_state_energies=np.concatenate(
                [bundle.train.ground_state_energies, bundle.test.ground_state_energies],
                axis=0,
            ),
        )

    def test_qiskit_pure_su2_forward_matches_numpy_backend(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.7)
        parameters = np.asarray((0.15, -0.1, 0.05), dtype=np.float64)

        numpy_model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters, backend=NumpyPureStateBackend())
        qiskit_model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters, backend=QiskitPureStateBackend())

        numpy_forward = numpy_model.forward(state)
        qiskit_forward = qiskit_model.forward(state)
        np.testing.assert_allclose(
            qiskit_forward.final_density_matrix,
            numpy_forward.final_density_matrix,
            atol=1e-10,
        )
        self.assertAlmostEqual(qiskit_forward.probability, numpy_forward.probability, places=10)

    def test_qiskit_pure_hea_loss_matches_numpy_backend(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=4)
        parameters = np.linspace(-0.2, 0.2, 24)

        numpy_model = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=parameters, backend=NumpyPureStateBackend())
        qiskit_model = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=parameters, backend=QiskitPureStateBackend())

        self.assertAlmostEqual(
            qiskit_model.loss(dataset.states, dataset.labels, loss_name="bce"),
            numpy_model.loss(dataset.states, dataset.labels, loss_name="bce"),
            places=10,
        )
        self.assertAlmostEqual(
            qiskit_model.loss(dataset.states, dataset.labels, loss_name="mse"),
            numpy_model.loss(dataset.states, dataset.labels, loss_name="mse"),
            places=10,
        )

    def test_qiskit_mixed_no_noise_forward_is_finite(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(1.1)
        model = SU2QCNN(
            QCNNConfig(num_qubits=4),
            parameters=np.asarray((0.05, -0.03, 0.08), dtype=np.float64),
            backend=QiskitMixedStateBackend(),
        )

        forward = model.forward(state)
        self.assertEqual(forward.final_density_matrix.shape, (4, 4))
        self.assertTrue(np.isfinite(forward.probability))
        self.assertGreaterEqual(forward.probability, 0.0)
        self.assertLessEqual(forward.probability, 1.0)

    def test_qiskit_mixed_noisy_runner_smoke_writes_artifacts(self) -> None:
        dataset = generate_dataset(HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=9))
        backend = QiskitMixedStateBackend(
            noise_config=NoiseConfig(
                noise_model_name="depolarizing",
                single_qubit_depolarizing_error=0.01,
                two_qubit_depolarizing_error=0.02,
                readout_error_probability=0.03,
            )
        )
        training_config = TrainingConfig(
            epochs=1,
            learning_rate=0.05,
            gradient_backend="finite_difference",
            random_seed=0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "qiskit_mixed_smoke"
            result = run_training_experiment(
                dataset,
                ExperimentConfig(model_family="hea_qcnn", backend_name="qiskit_mixed", num_qubits=4),
                training_config,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertTrue(np.isfinite(result["train_metrics"]["loss"]))
            self.assertTrue(np.isfinite(result["test_metrics"]["loss"]))
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "best_parameters.npy").exists())

    def test_runner_builds_qiskit_backends(self) -> None:
        self.assertIsInstance(build_backend("qiskit_pure"), QiskitPureStateBackend)
        self.assertIsInstance(build_backend("qiskit_mixed"), QiskitMixedStateBackend)


if __name__ == "__main__":
    unittest.main()
