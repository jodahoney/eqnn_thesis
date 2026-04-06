"""Qiskit-backed mixed-state backend with simple channel noise."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from eqnn.circuits.qiskit_builders import (
    active_pairs_for_convolution,
    active_qubits_for_convolution,
    qiskit_density_to_repo,
    repo_density_to_qiskit,
    repo_sites_to_qiskit,
)
from eqnn.models.base import QCNNForwardPass
from eqnn.noise import NoiseConfig

from eqnn.backends.qiskit_pure import QISKIT_AVAILABLE, QiskitPureStateBackend

try:  # pragma: no cover - exercised when qiskit is installed
    from qiskit.quantum_info import DensityMatrix, Kraus
except ImportError:  # pragma: no cover - local environments may not have qiskit
    DensityMatrix = None  # type: ignore[assignment]
    Kraus = None  # type: ignore[assignment]


_IDENTITY_1Q = np.eye(2, dtype=np.complex128)
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


class QiskitMixedStateBackend(QiskitPureStateBackend):
    """Mixed-state Qiskit backend with simple explicit noise channels."""

    def __init__(self, *, noise_config: NoiseConfig | None = None) -> None:
        super().__init__()
        if DensityMatrix is None or Kraus is None:  # pragma: no cover - guarded in tests
            raise ImportError(
                "QiskitMixedStateBackend requires the optional 'qiskit' dependency. "
                "Install it with `pip install 'eqnn-simulator[qiskit]'`."
            )
        self.noise_config = NoiseConfig() if noise_config is None else noise_config

    def _apply_post_convolution_noise(
        self,
        convolution: object,
        density_matrix: np.ndarray,
        num_qubits: int,
    ) -> np.ndarray:
        if self.noise_config.is_noiseless:
            return np.asarray(density_matrix, dtype=np.complex128)

        qiskit_density = DensityMatrix(repo_density_to_qiskit(np.asarray(density_matrix, dtype=np.complex128), num_qubits))
        active_qubits = active_qubits_for_convolution(convolution)
        active_pairs = active_pairs_for_convolution(convolution)

        if self.noise_config.noise_model_name == "depolarizing":
            if self.noise_config.single_qubit_depolarizing_error > 0.0:
                channel = Kraus(self._single_qubit_depolarizing_kraus(self.noise_config.single_qubit_depolarizing_error))
                for site in active_qubits:
                    qiskit_density = qiskit_density.evolve(channel, qargs=[repo_sites_to_qiskit(num_qubits, (site,))[0]])
            if self.noise_config.two_qubit_depolarizing_error > 0.0:
                channel = Kraus(self._two_qubit_depolarizing_kraus(self.noise_config.two_qubit_depolarizing_error))
                for pair in active_pairs:
                    qargs = list(repo_sites_to_qiskit(num_qubits, pair))
                    qiskit_density = qiskit_density.evolve(channel, qargs=qargs)

        if self.noise_config.noise_model_name == "amplitude_damping" and self.noise_config.amplitude_damping_gamma > 0.0:
            channel = Kraus(self._amplitude_damping_kraus(self.noise_config.amplitude_damping_gamma))
            for site in active_qubits:
                qiskit_density = qiskit_density.evolve(channel, qargs=[repo_sites_to_qiskit(num_qubits, (site,))[0]])

        if self.noise_config.noise_model_name == "phase_damping" and self.noise_config.phase_damping_gamma > 0.0:
            channel = Kraus(self._phase_damping_kraus(self.noise_config.phase_damping_gamma))
            for site in active_qubits:
                qiskit_density = qiskit_density.evolve(channel, qargs=[repo_sites_to_qiskit(num_qubits, (site,))[0]])

        return np.asarray(qiskit_density_to_repo(np.asarray(qiskit_density.data), num_qubits), dtype=np.complex128)

    def _postprocess_forward_pass(self, forward_pass: object) -> object:
        if not isinstance(forward_pass, QCNNForwardPass):
            return forward_pass
        corrected_probability = self.noise_config.apply_readout_error(forward_pass.probability)
        if corrected_probability == forward_pass.probability:
            return forward_pass
        return replace(forward_pass, probability=corrected_probability)

    @staticmethod
    def _single_qubit_depolarizing_kraus(probability: float) -> list[np.ndarray]:
        probability = float(probability)
        identity_weight = max(0.0, 1.0 - 0.75 * probability)
        pauli_weight = probability / 4.0
        return [
            np.sqrt(identity_weight) * _IDENTITY_1Q,
            np.sqrt(pauli_weight) * _PAULI_X,
            np.sqrt(pauli_weight) * _PAULI_Y,
            np.sqrt(pauli_weight) * _PAULI_Z,
        ]

    @staticmethod
    def _two_qubit_depolarizing_kraus(probability: float) -> list[np.ndarray]:
        probability = float(probability)
        paulis = (_IDENTITY_1Q, _PAULI_X, _PAULI_Y, _PAULI_Z)
        kraus_ops: list[np.ndarray] = [np.sqrt(max(0.0, 1.0 - probability)) * np.kron(_IDENTITY_1Q, _IDENTITY_1Q)]
        if probability <= 0.0:
            return kraus_ops
        non_identity_ops = [
            np.kron(left, right)
            for left in paulis
            for right in paulis
            if not (np.array_equal(left, _IDENTITY_1Q) and np.array_equal(right, _IDENTITY_1Q))
        ]
        error_weight = np.sqrt(probability / float(len(non_identity_ops)))
        kraus_ops.extend(error_weight * operator for operator in non_identity_ops)
        return kraus_ops

    @staticmethod
    def _amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
        gamma = float(gamma)
        return [
            np.asarray(((1.0, 0.0), (0.0, np.sqrt(1.0 - gamma))), dtype=np.complex128),
            np.asarray(((0.0, np.sqrt(gamma)), (0.0, 0.0)), dtype=np.complex128),
        ]

    @staticmethod
    def _phase_damping_kraus(gamma: float) -> list[np.ndarray]:
        gamma = float(gamma)
        return [
            np.asarray(((1.0, 0.0), (0.0, np.sqrt(1.0 - gamma))), dtype=np.complex128),
            np.asarray(((0.0, 0.0), (0.0, np.sqrt(gamma))), dtype=np.complex128),
        ]
