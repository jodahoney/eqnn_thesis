"""Numerical equivariance and invariance checks for SU(2)-aware components."""

from __future__ import annotations

import numpy as np

from eqnn.groups.su2 import SU2Group
from eqnn.physics.quantum import as_density_matrix


def random_complex_statevector(num_qubits: int, seed: int) -> np.ndarray:
    """Sample a normalized random n-qubit statevector."""

    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << num_qubits) + 1.0j * rng.normal(size=1 << num_qubits)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def random_su2_rotation(num_qubits: int, seed: int) -> np.ndarray:
    """Sample a reproducible global SU(2) rotation U^{⊗n}."""

    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    angle = rng.uniform(-np.pi, np.pi)
    return SU2Group().global_rotation(num_qubits, tuple(axis.tolist()), float(angle))


def convolution_equivariance_error(layer: object, num_trials: int = 10, seed: int = 0) -> dict[str, float]:
    """Check that a convolution layer commutes with the global SU(2) action."""

    errors = []
    for trial in range(num_trials):
        state = random_complex_statevector(layer.config.num_qubits, seed + trial)
        rotation = random_su2_rotation(layer.config.num_qubits, seed + 10_000 + trial)
        left = layer(rotation @ state)
        right = rotation @ layer(state)
        errors.append(float(np.linalg.norm(left - right)))
    return _summarize_errors(errors)


def convolution_operator_equivariance_error(
    layer: object,
    num_trials: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Check equivariance at the operator level via commutators."""

    unitary = layer.unitary()
    errors = []
    for trial in range(num_trials):
        rotation = random_su2_rotation(layer.config.num_qubits, seed + 15_000 + trial)
        commutator = unitary @ rotation - rotation @ unitary
        errors.append(float(np.linalg.norm(commutator)))
    return _summarize_errors(errors)


def pooling_equivariance_error(layer: object, num_trials: int = 10, seed: int = 0) -> dict[str, float]:
    """Check that pooling is equivariant under the global SU(2) action."""

    errors = []
    for trial in range(num_trials):
        state = random_complex_statevector(layer.config.num_qubits, seed + trial)
        density_matrix = as_density_matrix(state)
        input_rotation = random_su2_rotation(layer.config.num_qubits, seed + 20_000 + trial)
        output_rotation = random_su2_rotation(
            layer.output_num_qubits,
            seed + 20_000 + trial,
        )

        rotated_input = input_rotation @ density_matrix @ input_rotation.conjugate().T
        left = layer(rotated_input)
        right = output_rotation @ layer(density_matrix) @ output_rotation.conjugate().T
        errors.append(float(np.linalg.norm(left - right)))
    return _summarize_errors(errors)


def model_invariance_error(model: object, num_trials: int = 10, seed: int = 0) -> dict[str, float]:
    """Check that the QCNN scalar prediction is invariant under global SU(2)."""

    errors = []
    for trial in range(num_trials):
        state = random_complex_statevector(model.config.num_qubits, seed + trial)
        rotation = random_su2_rotation(model.config.num_qubits, seed + 30_000 + trial)
        errors.append(abs(model.predict(state) - model.predict(rotation @ state)))
    return _summarize_errors(errors)


def check_global_su2_equivariance(model: object, num_trials: int = 10) -> dict[str, float]:
    """Return a compact summary of the QCNN prediction invariance error."""

    return model_invariance_error(model, num_trials=num_trials)


def estimate_equivariance_error(
    model: object,
    states: np.ndarray,
    *,
    num_symmetry_samples: int = 8,
    seed: int | None = None,
    backend: object | None = None,
) -> dict[str, float | bool | str | int]:
    """Estimate empirical prediction drift under sampled global SU(2) transformations.

    This is intentionally lightweight: it measures the mean and max absolute
    prediction change under sampled global SU(2) rotations. If the model does
    not expose the expected prediction surface, the diagnostic reports that
    cleanly instead of fabricating a value. It is an empirical stability check,
    not a theoretical certificate of equivariance.
    """

    del backend
    if not hasattr(model, "predict"):
        return {
            "available": False,
            "note": "model does not expose predict(state)",
            "mean_error": 0.0,
            "max_error": 0.0,
            "num_state_samples": 0,
            "num_symmetry_samples": int(num_symmetry_samples),
        }
    if not hasattr(model, "config") or not hasattr(model.config, "num_qubits"):
        return {
            "available": False,
            "note": "model does not expose config.num_qubits",
            "mean_error": 0.0,
            "max_error": 0.0,
            "num_state_samples": 0,
            "num_symmetry_samples": int(num_symmetry_samples),
        }

    state_array = np.asarray(states, dtype=np.complex128)
    if state_array.ndim == 1:
        state_array = state_array[np.newaxis, :]
    if state_array.ndim != 2:
        return {
            "available": False,
            "note": "states must have shape (num_states, hilbert_dimension)",
            "mean_error": 0.0,
            "max_error": 0.0,
            "num_state_samples": 0,
            "num_symmetry_samples": int(num_symmetry_samples),
        }

    base_seed = 0 if seed is None else int(seed)
    errors: list[float] = []
    num_qubits = int(model.config.num_qubits)

    for state_index, state in enumerate(state_array):
        baseline = float(model.predict(state))
        for sample_index in range(int(num_symmetry_samples)):
            rotation = random_su2_rotation(
                num_qubits,
                base_seed + 10_000 * state_index + sample_index,
            )
            transformed_state = rotation @ state
            transformed_prediction = float(model.predict(transformed_state))
            errors.append(abs(baseline - transformed_prediction))

    if not errors:
        return {
            "available": False,
            "note": "no states were provided",
            "mean_error": 0.0,
            "max_error": 0.0,
            "num_state_samples": 0,
            "num_symmetry_samples": int(num_symmetry_samples),
        }

    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "available": True,
        "note": "empirical_prediction_drift_under_sampled_global_su2",
        "mean_error": float(np.mean(error_array)),
        "max_error": float(np.max(error_array)),
        "num_state_samples": int(state_array.shape[0]),
        "num_symmetry_samples": int(num_symmetry_samples),
    }


def _summarize_errors(errors: list[float]) -> dict[str, float]:
    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "max_error": float(np.max(error_array)),
        "mean_error": float(np.mean(error_array)),
        "std_error": float(np.std(error_array)),
    }
