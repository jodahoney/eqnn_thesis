from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eqnn.backends import NumpyPureStateBackend, TORCH_AVAILABLE, TorchPureStateBackend
from eqnn.backends.torch_ops import (
    partial_trace_density_matrix,
    statevector_to_density_matrix,
    statevectors_to_density_matrices,
)
from eqnn.datasets.heisenberg import DatasetSplit, HeisenbergDatasetConfig, generate_dataset
from eqnn.experiments.runner import ExperimentConfig, run_training_experiment
from eqnn.models import (
    BaselineQCNN,
    BaselineQCNNConfig,
    HEAQCNN,
    HEAQCNNConfig,
    QCNNConfig,
    SU2QCNN,
)
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.training import Trainer, TrainingConfig


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchPureStateBackendParityTests(unittest.TestCase):
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

    def _assert_forward_parity(self, numpy_model: object, torch_model: object, state: np.ndarray) -> None:
        numpy_forward = numpy_model.forward(state)
        torch_forward = torch_model.forward(state)
        np.testing.assert_allclose(
            torch_forward.final_density_matrix,
            numpy_forward.final_density_matrix,
            atol=1e-10,
        )
        self.assertAlmostEqual(torch_forward.probability, numpy_forward.probability, places=10)

    def test_su2_forward_matches_numpy_backend(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.6)
        parameters = np.asarray((0.2, -0.15, 0.1), dtype=np.float64)

        numpy_model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters, backend=NumpyPureStateBackend())
        torch_model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters, backend=TorchPureStateBackend())

        self._assert_forward_parity(numpy_model, torch_model, state)

    def test_hea_forward_matches_numpy_backend(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.8)
        parameters = np.linspace(-0.2, 0.2, 24)

        numpy_model = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=parameters, backend=NumpyPureStateBackend())
        torch_model = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=parameters, backend=TorchPureStateBackend())

        self._assert_forward_parity(numpy_model, torch_model, state)

    def test_baseline_forward_matches_numpy_backend(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(1.2)
        parameters = np.asarray((0.2, -0.4, 0.1, 0.3, 0.05, -0.2, -0.1, 0.25, 0.15), dtype=np.float64)

        numpy_model = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4),
            parameters=parameters,
            backend=NumpyPureStateBackend(),
        )
        torch_model = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4),
            parameters=parameters,
            backend=TorchPureStateBackend(),
        )

        self._assert_forward_parity(numpy_model, torch_model, state)

    def test_loss_matches_numpy_backend_for_bce_and_mse(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=7)
        parameters = np.asarray((0.09, -0.03, 0.14), dtype=np.float64)

        numpy_model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters, backend=NumpyPureStateBackend())
        torch_model = SU2QCNN(QCNNConfig(num_qubits=4), parameters=parameters, backend=TorchPureStateBackend())

        self.assertAlmostEqual(
            torch_model.loss(dataset.states, dataset.labels, loss_name="bce"),
            numpy_model.loss(dataset.states, dataset.labels, loss_name="bce"),
            places=10,
        )
        self.assertAlmostEqual(
            torch_model.loss(dataset.states, dataset.labels, loss_name="mse"),
            numpy_model.loss(dataset.states, dataset.labels, loss_name="mse"),
            places=10,
        )

    def test_predict_batch_matches_numpy_backend(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=10)

        su2_parameters = np.asarray((0.18, -0.06, 0.12), dtype=np.float64)
        numpy_su2 = SU2QCNN(QCNNConfig(num_qubits=4), parameters=su2_parameters, backend=NumpyPureStateBackend())
        torch_su2 = SU2QCNN(QCNNConfig(num_qubits=4), parameters=su2_parameters, backend=TorchPureStateBackend())
        np.testing.assert_allclose(
            torch_su2.predict_batch(dataset.states),
            numpy_su2.predict_batch(dataset.states),
            atol=1e-10,
        )

        hea_parameters = np.linspace(-0.2, 0.2, 24)
        numpy_hea = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=hea_parameters, backend=NumpyPureStateBackend())
        torch_hea = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=hea_parameters, backend=TorchPureStateBackend())
        np.testing.assert_allclose(
            torch_hea.predict_batch(dataset.states),
            numpy_hea.predict_batch(dataset.states),
            atol=1e-10,
        )

        baseline_parameters = np.asarray((0.2, -0.4, 0.1, 0.3, 0.05, -0.2, -0.1, 0.25, 0.15), dtype=np.float64)
        numpy_baseline = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4),
            parameters=baseline_parameters,
            backend=NumpyPureStateBackend(),
        )
        torch_baseline = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4),
            parameters=baseline_parameters,
            backend=TorchPureStateBackend(),
        )
        np.testing.assert_allclose(
            torch_baseline.predict_batch(dataset.states),
            numpy_baseline.predict_batch(dataset.states),
            atol=1e-10,
        )

    def test_batched_partial_trace_matches_stacked_unbatched_results(self) -> None:
        import torch

        zero_zero = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.complex128)
        singlet = torch.tensor(
            (0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0),
            dtype=torch.complex128,
        )
        states = torch.stack((zero_zero, singlet), dim=0)

        batched_density = statevectors_to_density_matrices(states)
        batched_reduced = partial_trace_density_matrix(
            batched_density,
            num_qubits=2,
            traced_out_sites=(1,),
        )

        expected = torch.stack(
            [
                partial_trace_density_matrix(
                    statevector_to_density_matrix(state),
                    num_qubits=2,
                    traced_out_sites=(1,),
                )
                for state in states
            ],
            dim=0,
        )

        torch.testing.assert_close(batched_reduced, expected)

    def test_trainer_evaluate_matches_numpy_backend(self) -> None:
        dataset_bundle = generate_dataset(HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=11))
        trainer = Trainer(TrainingConfig(loss="mse", gradient_backend="exact"))
        parameters = np.linspace(-0.15, 0.25, 24)

        numpy_model = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=parameters, backend=NumpyPureStateBackend())
        torch_model = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=parameters, backend=TorchPureStateBackend())

        numpy_metrics = trainer.evaluate(numpy_model, dataset_bundle.test)
        torch_metrics = trainer.evaluate(torch_model, dataset_bundle.test)

        self.assertAlmostEqual(torch_metrics["loss"], numpy_metrics["loss"], places=10)
        self.assertAlmostEqual(torch_metrics["accuracy"], numpy_metrics["accuracy"], places=10)

    def test_gradients_match_numpy_backend(self) -> None:
        dataset = self._combined_dataset(num_qubits=4, num_points=5, split_seed=8)

        su2_parameters = np.asarray((0.12, -0.08, 0.17), dtype=np.float64)
        numpy_su2 = SU2QCNN(QCNNConfig(num_qubits=4), parameters=su2_parameters, backend=NumpyPureStateBackend())
        torch_su2 = SU2QCNN(QCNNConfig(num_qubits=4), parameters=su2_parameters, backend=TorchPureStateBackend())
        np.testing.assert_allclose(
            torch_su2.loss_gradient(dataset.states, dataset.labels, loss_name="bce"),
            numpy_su2.loss_gradient(dataset.states, dataset.labels, loss_name="bce"),
            atol=1e-6,
        )

        hea_parameters = np.linspace(-0.15, 0.25, 24)
        numpy_hea = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=hea_parameters, backend=NumpyPureStateBackend())
        torch_hea = HEAQCNN(HEAQCNNConfig(num_qubits=4), parameters=hea_parameters, backend=TorchPureStateBackend())
        np.testing.assert_allclose(
            torch_hea.loss_gradient(dataset.states, dataset.labels, loss_name="bce"),
            numpy_hea.loss_gradient(dataset.states, dataset.labels, loss_name="bce"),
            atol=1e-6,
        )

        baseline_parameters = np.asarray((0.15, -0.11, 0.07, 0.05, 0.09, -0.13, 0.8, -0.2), dtype=np.float64)
        numpy_baseline = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4, min_readout_qubits=4, readout_mode="dimerization"),
            parameters=baseline_parameters,
            backend=NumpyPureStateBackend(),
        )
        torch_baseline = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4, min_readout_qubits=4, readout_mode="dimerization"),
            parameters=baseline_parameters,
            backend=TorchPureStateBackend(),
        )
        np.testing.assert_allclose(
            torch_baseline.loss_gradient(dataset.states, dataset.labels, loss_name="mse"),
            numpy_baseline.loss_gradient(dataset.states, dataset.labels, loss_name="mse"),
            atol=1e-6,
        )

    def test_runner_smoke_with_torch_backend_writes_artifacts(self) -> None:
        dataset = generate_dataset(HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=9))
        training_config = TrainingConfig(
            epochs=1,
            learning_rate=0.05,
            gradient_backend="exact",
            random_seed=0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "torch_runner_smoke"
            result = run_training_experiment(
                dataset,
                ExperimentConfig(model_family="hea_qcnn", num_qubits=4),
                training_config,
                output_dir=output_dir,
                backend=TorchPureStateBackend(),
            )

            self.assertIn("train_metrics", result)
            self.assertIn("test_metrics", result)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "best_parameters.npy").exists())


if __name__ == "__main__":
    unittest.main()
