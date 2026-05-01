"""Numerical equivariance and invariance checks for SU(2)-aware components."""

from __future__ import annotations

from typing import Any

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


def evaluate_with_symmetry_twirling(
    model: object,
    states: np.ndarray,
    parameters: np.ndarray | None = None,
    backend: object | None = None,
    *,
    labels: np.ndarray | None = None,
    threshold: float | None = None,
    num_symmetry_samples: int = 8,
    seed: int | None = None,
) -> dict[str, Any]:
    """Average predictions over sampled global SU(2)-transformed inputs.

    This is an evaluation-time error mitigation helper. It does not change the
    trained model or implement encoded quantum error correction; it simply
    evaluates f(g rho g^dagger) for sampled global rotations and averages the
    resulting scalar predictions.
    """

    del backend
    if int(num_symmetry_samples) < 1:
        raise ValueError("num_symmetry_samples must be at least 1")

    unavailable = _symmetry_twirling_unavailable_result(
        note="model does not expose predict(state)",
        num_symmetry_samples=int(num_symmetry_samples),
    )
    if not hasattr(model, "predict"):
        return unavailable

    try:
        state_list, state_kind = _coerce_state_list(states, model)
        if not state_list:
            return _symmetry_twirling_unavailable_result(
                note="no states were provided",
                num_symmetry_samples=int(num_symmetry_samples),
            )
        num_qubits = _resolve_num_qubits(model, state_list[0])
        threshold_value = _resolve_threshold(model, threshold)
        label_array = None if labels is None else np.asarray(labels, dtype=np.int64)
        if label_array is not None and label_array.shape != (len(state_list),):
            return _symmetry_twirling_unavailable_result(
                note="labels must align with states",
                num_symmetry_samples=int(num_symmetry_samples),
            )

        base_seed = 0 if seed is None else int(seed)
        raw_probabilities: list[float] = []
        twirled_probabilities: list[float] = []

        for state_index, state in enumerate(state_list):
            raw_probability = _predict_probability(model, state, parameters)
            raw_probabilities.append(raw_probability)

            transformed_probabilities: list[float] = []
            for sample_index in range(int(num_symmetry_samples)):
                rotation = random_su2_rotation(
                    num_qubits,
                    base_seed + 10_000 * state_index + sample_index,
                )
                transformed_state = _apply_global_rotation(state, rotation, state_kind)
                transformed_probabilities.append(_predict_probability(model, transformed_state, parameters))
            twirled_probabilities.append(float(np.mean(transformed_probabilities)))

        raw_array = np.asarray(raw_probabilities, dtype=np.float64)
        twirled_array = np.asarray(twirled_probabilities, dtype=np.float64)
        result: dict[str, Any] = {
            "available": True,
            "note": "symmetry_twirled_prediction_average_under_sampled_global_su2",
            "symmetry_twirling_available": True,
            "symmetry_twirling_note": "symmetry_twirled_prediction_average_under_sampled_global_su2",
            "raw_probabilities": raw_array,
            "twirled_probabilities": twirled_array,
            "raw_accuracy": None,
            "twirled_accuracy": None,
            "num_correct_raw": None,
            "num_correct_twirled": None,
            "mean_abs_twirling_shift": float(np.mean(np.abs(twirled_array - raw_array))),
            "num_state_samples": int(len(state_list)),
            "num_symmetry_samples": int(num_symmetry_samples),
            "symmetry_twirled_raw_subset_accuracy": None,
            "symmetry_twirled_test_accuracy": None,
            "symmetry_twirled_subset_size": int(len(state_list)),
            "symmetry_twirled_num_correct_raw_subset": None,
            "symmetry_twirled_num_correct_twirled_subset": None,
        }
        if label_array is not None:
            raw_predictions = (raw_array >= threshold_value).astype(np.int64)
            twirled_predictions = (twirled_array >= threshold_value).astype(np.int64)
            num_correct_raw = int(np.sum(raw_predictions == label_array))
            num_correct_twirled = int(np.sum(twirled_predictions == label_array))
            raw_accuracy = float(num_correct_raw / int(label_array.size))
            twirled_accuracy = float(num_correct_twirled / int(label_array.size))
            result["raw_accuracy"] = raw_accuracy
            result["twirled_accuracy"] = twirled_accuracy
            result["num_correct_raw"] = num_correct_raw
            result["num_correct_twirled"] = num_correct_twirled
            result["symmetry_twirled_raw_subset_accuracy"] = raw_accuracy
            result["symmetry_twirled_test_accuracy"] = twirled_accuracy
            result["symmetry_twirled_num_correct_raw_subset"] = num_correct_raw
            result["symmetry_twirled_num_correct_twirled_subset"] = num_correct_twirled
        return result
    except Exception as exc:  # pragma: no cover - defensive path for optional evaluation
        return _symmetry_twirling_unavailable_result(
            note=f"not_available: {type(exc).__name__}: {exc}",
            num_symmetry_samples=int(num_symmetry_samples),
        )


def _coerce_state_list(states: np.ndarray, model: object) -> tuple[list[np.ndarray], str]:
    state_array = np.asarray(states, dtype=np.complex128)
    num_qubits = getattr(getattr(model, "config", None), "num_qubits", None)
    hilbert_dimension = None if num_qubits is None else 1 << int(num_qubits)

    if state_array.ndim == 1:
        return [state_array], "statevector"
    if state_array.ndim == 2:
        if hilbert_dimension is not None and state_array.shape[1] != hilbert_dimension:
            raise ValueError("statevector width does not match model.config.num_qubits")
        return [np.asarray(state, dtype=np.complex128) for state in state_array], "statevector"
    if state_array.ndim == 3 and state_array.shape[1] == state_array.shape[2]:
        if hilbert_dimension is not None and state_array.shape[1] != hilbert_dimension:
            raise ValueError("density matrix dimension does not match model.config.num_qubits")
        return [np.asarray(state, dtype=np.complex128) for state in state_array], "density_matrix"
    raise ValueError("states must be statevectors with shape (num_states, dim) or density matrices with ndim=3")


def _resolve_num_qubits(model: object, state: np.ndarray) -> int:
    if hasattr(model, "config") and hasattr(model.config, "num_qubits"):
        return int(model.config.num_qubits)
    dimension = state.shape[0]
    num_qubits = int(round(np.log2(dimension)))
    if 1 << num_qubits != int(dimension):
        raise ValueError("could not infer num_qubits from state dimension")
    return num_qubits


def _resolve_threshold(model: object, threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    if hasattr(model, "get_classification_threshold"):
        return float(model.get_classification_threshold())
    return 0.5


def _predict_probability(model: object, state: np.ndarray, parameters: np.ndarray | None) -> float:
    if parameters is None:
        return float(model.predict(state))
    try:
        return float(model.predict(state, parameters=parameters))
    except TypeError:
        return float(model.predict(state))


def _apply_global_rotation(state: np.ndarray, rotation: np.ndarray, state_kind: str) -> np.ndarray:
    if state_kind == "density_matrix":
        return np.asarray(rotation @ state @ rotation.conjugate().T, dtype=np.complex128)
    return np.asarray(rotation @ state, dtype=np.complex128)


def _symmetry_twirling_unavailable_result(
    *,
    note: str,
    num_symmetry_samples: int,
) -> dict[str, Any]:
    empty = np.asarray([], dtype=np.float64)
    return {
        "available": False,
        "note": note,
        "symmetry_twirling_available": False,
        "symmetry_twirling_note": note,
        "raw_probabilities": empty,
        "twirled_probabilities": empty,
        "raw_accuracy": None,
        "twirled_accuracy": None,
        "num_correct_raw": None,
        "num_correct_twirled": None,
        "mean_abs_twirling_shift": None,
        "num_state_samples": 0,
        "num_symmetry_samples": int(num_symmetry_samples),
        "symmetry_twirled_raw_subset_accuracy": None,
        "symmetry_twirled_test_accuracy": None,
        "symmetry_twirled_subset_size": 0,
        "symmetry_twirled_num_correct_raw_subset": None,
        "symmetry_twirled_num_correct_twirled_subset": None,
    }


def _summarize_errors(errors: list[float]) -> dict[str, float]:
    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "max_error": float(np.max(error_array)),
        "mean_error": float(np.mean(error_array)),
        "std_error": float(np.std(error_array)),
    }
