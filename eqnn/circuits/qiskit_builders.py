"""Small Qiskit circuit builders for QCNN backend integrations."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from eqnn.layers import AnisotropicConvolution, HEAConvolution, SU2SwapConvolution
from eqnn.physics.observables import SWAP_OPERATOR

try:  # pragma: no cover - exercised when qiskit is installed
    from qiskit import QuantumCircuit
except ImportError:  # pragma: no cover - local environments may not have qiskit
    QuantumCircuit = None  # type: ignore[assignment]

try:  # pragma: no cover - import location differs slightly across qiskit releases
    from qiskit.circuit.library import UnitaryGate
except ImportError:  # pragma: no cover
    try:
        from qiskit.extensions import UnitaryGate  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        UnitaryGate = None  # type: ignore[assignment]


QISKIT_AVAILABLE = QuantumCircuit is not None and UnitaryGate is not None


def require_qiskit() -> None:
    if not QISKIT_AVAILABLE:  # pragma: no cover - guarded by QISKIT_AVAILABLE in tests
        raise ImportError(
            "Qiskit backends require the optional 'qiskit' dependency. "
            "Install it with `pip install 'eqnn-simulator[qiskit]'`."
        )


def build_state_preparation_circuit(statevector: np.ndarray) -> "QuantumCircuit":
    require_qiskit()
    state = np.asarray(statevector, dtype=np.complex128)
    if state.ndim != 1:
        raise ValueError("statevector must be one-dimensional")
    num_qubits = _num_qubits_from_dimension(state.shape[0])
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(repo_statevector_to_qiskit(state, num_qubits), list(range(num_qubits)))
    return circuit


def build_swap_readout_operator() -> np.ndarray:
    return np.asarray(SWAP_OPERATOR, dtype=np.complex128)


def build_su2_pair_block(theta: float) -> "QuantumCircuit":
    require_qiskit()
    circuit = QuantumCircuit(2)
    circuit.append(UnitaryGate(SU2SwapConvolution.gate(float(theta)), label="su2_swap"), [0, 1])
    return circuit


def build_hea_pair_block(parameters: Iterable[float]) -> "QuantumCircuit":
    require_qiskit()
    parameter_array = np.asarray(list(parameters), dtype=np.float64)
    if parameter_array.shape != (8,):
        raise ValueError("HEA pair blocks expect 8 parameters")
    circuit = QuantumCircuit(2)
    circuit.ry(float(parameter_array[0]), 0)
    circuit.rz(float(parameter_array[1]), 0)
    circuit.ry(float(parameter_array[2]), 1)
    circuit.rz(float(parameter_array[3]), 1)
    circuit.cz(0, 1)
    circuit.ry(float(parameter_array[4]), 0)
    circuit.rz(float(parameter_array[5]), 0)
    circuit.ry(float(parameter_array[6]), 1)
    circuit.rz(float(parameter_array[7]), 1)
    return circuit


def build_anisotropic_pair_block(parameters: Iterable[float]) -> "QuantumCircuit":
    require_qiskit()
    parameter_array = np.asarray(list(parameters), dtype=np.float64)
    if parameter_array.shape != (3,):
        raise ValueError("Anisotropic pair blocks expect 3 parameters")
    circuit = QuantumCircuit(2)
    circuit.append(
        UnitaryGate(
            AnisotropicConvolution.gate(
                float(parameter_array[0]),
                float(parameter_array[1]),
                float(parameter_array[2]),
            ),
            label="anisotropic",
        ),
        [0, 1],
    )
    return circuit


def build_convolution_circuit(convolution: object, parameters: Iterable[float]) -> "QuantumCircuit":
    require_qiskit()
    parameter_array = np.asarray(list(parameters), dtype=np.float64)
    if not hasattr(convolution, "unitary"):
        raise TypeError("Convolution object must expose a unitary(...) method for Qiskit circuit construction")

    num_qubits = int(convolution.config.num_qubits)
    unitary_repo = np.asarray(convolution.unitary(parameters=parameter_array), dtype=np.complex128)
    unitary_qiskit = repo_density_to_qiskit(unitary_repo, num_qubits)

    circuit = QuantumCircuit(num_qubits)
    circuit.append(UnitaryGate(unitary_qiskit, label=type(convolution).__name__), list(range(num_qubits)))
    return circuit


def repo_statevector_to_qiskit(statevector: np.ndarray, num_qubits: int | None = None) -> np.ndarray:
    state = np.asarray(statevector, dtype=np.complex128)
    if state.ndim != 1:
        raise ValueError("statevector must be one-dimensional")
    resolved_qubits = _num_qubits_from_dimension(state.shape[0]) if num_qubits is None else int(num_qubits)
    tensor = state.reshape((2,) * resolved_qubits)
    return np.transpose(tensor, axes=tuple(reversed(range(resolved_qubits)))).reshape(-1)


def qiskit_statevector_to_repo(statevector: np.ndarray, num_qubits: int | None = None) -> np.ndarray:
    return repo_statevector_to_qiskit(statevector, num_qubits=num_qubits)


def repo_density_to_qiskit(density_matrix: np.ndarray, num_qubits: int | None = None) -> np.ndarray:
    density = np.asarray(density_matrix, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density_matrix must be square")
    resolved_qubits = _num_qubits_from_dimension(density.shape[0]) if num_qubits is None else int(num_qubits)
    tensor = density.reshape((2,) * (2 * resolved_qubits))
    axes = tuple(reversed(range(resolved_qubits))) + tuple(
        reversed(range(resolved_qubits, 2 * resolved_qubits))
    )
    return np.transpose(tensor, axes=axes).reshape(density.shape)


def qiskit_density_to_repo(density_matrix: np.ndarray, num_qubits: int | None = None) -> np.ndarray:
    return repo_density_to_qiskit(density_matrix, num_qubits=num_qubits)


def repo_sites_to_qiskit(num_qubits: int, sites: Iterable[int]) -> tuple[int, ...]:
    resolved = tuple(int(site) for site in sites)
    return tuple(num_qubits - 1 - site for site in resolved)


def active_pairs_for_convolution(convolution: object) -> tuple[tuple[int, int], ...]:
    if not hasattr(convolution, "active_parities") or not hasattr(convolution, "pairs_for_parity"):
        return tuple()
    pairs: list[tuple[int, int]] = []
    for parity in convolution.active_parities():
        pairs.extend(tuple(pair) for pair in convolution.pairs_for_parity(parity))
    return tuple(pairs)


def active_qubits_for_convolution(convolution: object) -> tuple[int, ...]:
    qubits = sorted({site for pair in active_pairs_for_convolution(convolution) for site in pair})
    return tuple(int(site) for site in qubits)


def _num_qubits_from_dimension(dimension: int) -> int:
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("dimension must be a positive power of two")
    return int(np.log2(dimension))
