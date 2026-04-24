"""Qiskit-backed mixed-state backend with simple channel noise."""

from __future__ import annotations

from dataclasses import replace
from math import cos, sin

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
from eqnn.utils.timing import timed

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
_IDENTITY_2Q = np.eye(4, dtype=np.complex128)
_ZZ = np.kron(_PAULI_Z, _PAULI_Z)


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
        self._overrotation_rng: np.random.Generator | None = None

    def forward(
        self,
        model: object,
        state: np.ndarray,
        parameters: np.ndarray,
    ) -> QCNNForwardPass:
        if self.noise_config.noise_model_name == "coherent_overrotation":
            base_seed = 0 if self.noise_config.coherent_overrotation_seed is None else int(
                self.noise_config.coherent_overrotation_seed
            )
            self._overrotation_rng = np.random.default_rng(base_seed)
        try:
            return super().forward(model, state, parameters)
        finally:
            self._overrotation_rng = None

    def _apply_post_convolution_noise(
        self,
        convolution: object,
        density_matrix: np.ndarray,
        num_qubits: int,
    ) -> np.ndarray:
        if self.noise_config.is_noiseless:
            return np.asarray(density_matrix, dtype=np.complex128)

        with timed(self.runtime_profile, "backend.qiskit.apply_noise"):
            qiskit_density = DensityMatrix(
                repo_density_to_qiskit(np.asarray(density_matrix, dtype=np.complex128), num_qubits)
            )
            active_qubits = active_qubits_for_convolution(convolution)
            active_pairs = active_pairs_for_convolution(convolution)
            target_sites = self._target_noise_sites(active_qubits, num_qubits)

            if self.noise_config.noise_model_name == "depolarizing":
                for site in target_sites:
                    strength = self._single_qubit_strength_for_site(
                        site,
                        self.noise_config.single_qubit_depolarizing_error,
                    )
                    if strength <= 0.0:
                        continue
                    channel = Kraus(self._single_qubit_depolarizing_kraus(strength))
                    qiskit_density = qiskit_density.evolve(
                        channel,
                        qargs=[repo_sites_to_qiskit(num_qubits, (site,))[0]],
                    )
                if self.noise_config.two_qubit_depolarizing_error > 0.0:
                    channel = Kraus(
                        self._two_qubit_depolarizing_kraus(self.noise_config.two_qubit_depolarizing_error)
                    )
                    for pair in active_pairs:
                        qargs = list(repo_sites_to_qiskit(num_qubits, pair))
                        qiskit_density = qiskit_density.evolve(
                            channel,
                            qargs=qargs,
                        )

            if (
                self.noise_config.noise_model_name == "amplitude_damping"
                and (
                    self.noise_config.amplitude_damping_gamma > 0.0
                    or self.noise_config.single_qubit_error_profile is not None
                )
            ):
                for site in target_sites:
                    strength = self._single_qubit_strength_for_site(site, self.noise_config.amplitude_damping_gamma)
                    if strength <= 0.0:
                        continue
                    channel = Kraus(self._amplitude_damping_kraus(strength))
                    qiskit_density = qiskit_density.evolve(
                        channel,
                        qargs=[repo_sites_to_qiskit(num_qubits, (site,))[0]],
                    )

            if (
                self.noise_config.noise_model_name == "phase_damping"
                and (
                    self.noise_config.phase_damping_gamma > 0.0
                    or self.noise_config.single_qubit_error_profile is not None
                )
            ):
                for site in target_sites:
                    strength = self._single_qubit_strength_for_site(site, self.noise_config.phase_damping_gamma)
                    if strength <= 0.0:
                        continue
                    channel = Kraus(self._phase_damping_kraus(strength))
                    qiskit_density = qiskit_density.evolve(
                        channel,
                        qargs=[repo_sites_to_qiskit(num_qubits, (site,))[0]],
                    )

            if (
                self.noise_config.noise_model_name == "coherent_overrotation"
                and not self.noise_config._coherent_noise_is_noiseless()
            ):
                for pair_index, pair in enumerate(active_pairs):
                    angle = self._resolve_overrotation_angle(pair, pair_index)
                    if angle == 0.0:
                        continue
                    channel = Kraus(
                        [
                            self._coherent_overrotation_unitary(
                                angle,
                                axis=self.noise_config.coherent_overrotation_axis,
                            )
                        ]
                    )
                    qargs = list(repo_sites_to_qiskit(num_qubits, pair))
                    qiskit_density = qiskit_density.evolve(channel, qargs=qargs)

            return np.asarray(qiskit_density_to_repo(np.asarray(qiskit_density.data), num_qubits), dtype=np.complex128)

    def _target_noise_sites(
        self,
        active_qubits: tuple[int, ...] | list[int],
        num_qubits: int,
    ) -> tuple[int, ...]:
        if self.noise_config.noise_application_scope == "active":
            return tuple(int(site) for site in active_qubits)
        if self.noise_config.noise_application_scope == "all":
            return tuple(range(num_qubits))
        selected = tuple(int(site) for site in (self.noise_config.noisy_qubits or ()))
        # In selected_qubits mode indices refer to the current effective register.
        # Pooling can shrink the register, so selected indices that are no longer
        # present are skipped instead of treated as invalid for later layers.
        return tuple(site for site in selected if site < num_qubits)

    def _single_qubit_strength_for_site(self, site: int, default_strength: float) -> float:
        # Per-qubit profiles override the scalar channel strength rather than scale it.
        profile = self.noise_config.single_qubit_error_profile
        if profile is not None and site < len(profile):
            return float(profile[site])
        return float(default_strength)

    def _resolve_overrotation_angle(
        self,
        pair: tuple[int, int],
        pair_index: int,
        layer_index: int | None = None,
    ) -> float:
        if self.noise_config.coherent_overrotation_axis != "zz":
            raise ValueError("coherent_overrotation_axis must currently be 'zz'")

        base_angle = float(self.noise_config.coherent_overrotation_angle)
        mode = self.noise_config.coherent_overrotation_mode

        if mode == "fixed":
            return base_angle
        if mode == "pair_dependent":
            angles = self.noise_config.pair_dependent_overrotation_angles
            # Pair-dependent angles are indexed by the active-pair order in the current
            # convolution layer, not by a physical qubit-pair label map.
            if angles is not None and pair_index < len(angles):
                return float(angles[pair_index])
            return base_angle
        rng = self._overrotation_rng
        if rng is None:
            rng = np.random.default_rng(self._overrotation_seed(pair, pair_index, layer_index))

        if mode == "stochastic":
            if rng.random() < float(self.noise_config.coherent_overrotation_probability):
                return base_angle
            return 0.0
        if mode == "random_angle":
            return float(rng.normal(loc=base_angle, scale=float(self.noise_config.coherent_overrotation_angle_std)))

        raise ValueError(
            "coherent_overrotation_mode must be 'fixed', 'stochastic', 'random_angle', or 'pair_dependent'"
        )

    def _overrotation_seed(
        self,
        pair: tuple[int, int],
        pair_index: int,
        layer_index: int | None,
    ) -> int:
        base_seed = 0 if self.noise_config.coherent_overrotation_seed is None else int(
            self.noise_config.coherent_overrotation_seed
        )
        layer_term = 0 if layer_index is None else int(layer_index) + 1
        return int(
            (
                base_seed
                + 104_729 * (pair[0] + 1)
                + 130_363 * (pair[1] + 1)
                + 161_803 * (pair_index + 1)
                + 199_999 * layer_term
            )
            % (2**63 - 1)
        )

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

    @staticmethod
    def _coherent_overrotation_unitary(angle: float, *, axis: str) -> np.ndarray:
        if axis != "zz":
            raise ValueError("coherent_overrotation_axis must currently be 'zz'")
        half_angle = 0.5 * float(angle)
        return np.asarray(cos(half_angle) * _IDENTITY_2Q - 1.0j * sin(half_angle) * _ZZ, dtype=np.complex128)
