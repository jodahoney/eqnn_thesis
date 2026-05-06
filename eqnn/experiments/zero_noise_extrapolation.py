"""Post-hoc zero-noise extrapolation helpers for noisy comparison outputs."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eqnn.experiments.noisy_comparison import load_completed_noisy_comparison_runs


DEFAULT_LOW_NOISE_MAX = 0.05
DEFAULT_CHANCE_ACCURACY = 0.5
SUPPORTED_FIT_TYPES = ("linear", "quadratic", "log_margin_linear")
SUPPORTED_FIT_RANGES = ("full", "low_noise")

_POLYNOMIAL_FIT_DEGREES = {"linear": 1, "quadratic": 2}
_GROUP_METADATA_FIELDS = (
    "backend_name",
    "model_family",
    "num_qubits",
    "train_size",
    "epochs",
    "seed",
    "mitigation_method",
    "noise_model_name",
    "expected_symmetry_breaking",
    "expected_symmetry_breaking_note",
    "noise_application_scope",
    "noisy_qubit_index",
    "noisy_qubits",
    "selected_noisy_qubit_pattern",
    "coherent_overrotation_mode",
    "noise_aware_training",
    "training_noise_strengths",
    "train_noise_strength_values",
    "training_noise_sampling",
    "train_noise_sampling_mode",
    "train_noise_includes_zero",
    "training_noise_seed",
    "symmetry_regularization",
    "symmetry_regularization_enabled",
    "symmetry_regularization_beta",
    "symmetry_regularization_weight",
    "num_symmetry_regularization_samples",
    "symmetry_regularization_frequency",
    "symmetry_regularization_state_samples",
    "symmetry_regularization_seed",
)
_PREFERRED_CSV_FIELDS = (
    "backend_name",
    "model_family",
    "num_qubits",
    "train_size",
    "epochs",
    "seed",
    "noise_model_name",
    "mitigation_method",
    "expected_symmetry_breaking",
    "expected_symmetry_breaking_note",
    "noise_application_scope",
    "noisy_qubit_index",
    "noisy_qubits",
    "selected_noisy_qubit_pattern",
    "coherent_overrotation_mode",
    "metric_name",
    "fit_type",
    "fit_range",
    "low_noise_max",
    "chance_accuracy",
    "num_points_total",
    "num_points_used",
    "noise_strengths_used",
    "observed_metric_values_used",
    "zero_noise_estimate",
    "fit_valid",
    "fit_warning",
    "fit_coefficients",
    "r_squared",
    "fit_target_space",
    "transformed_metric_values_used",
    "num_points_excluded_by_log_margin",
    "residual_mse",
    "residual_mse_space",
    "zne_metric_name",
    "zne_fit_type",
    "zne_fit_range",
    "zne_estimate",
    "zne_slope",
    "zne_intercept",
    "zne_num_points",
    "zne_noise_strengths_used",
    "zne_max_noise_strength",
    "zne_residual_mse",
    "zne_quadratic_coeff",
    "zne_linear_coeff",
    "log_margin_intercept",
    "log_margin_slope",
)


def _numpy_rank_warning_type() -> type[Warning] | None:
    rank_warning = getattr(np, "RankWarning", None)
    if rank_warning is not None:
        return rank_warning
    try:
        from numpy.exceptions import RankWarning

        return RankWarning
    except Exception:
        return None


def fit_zero_noise_extrapolation(
    rows: list[dict[str, Any]],
    *,
    metric_name: str = "test_accuracy",
    fit_type: str | None = "linear",
    fit_types: Sequence[str] | None = None,
    fit_ranges: Sequence[str] | None = None,
    max_noise_strength: float | None = None,
    noise_strengths: Sequence[float] | None = None,
    low_noise_max: float | None = None,
    chance_accuracy: float = DEFAULT_CHANCE_ACCURACY,
    min_points: int | None = None,
) -> list[dict[str, Any]]:
    """Fit zero-noise extrapolations grouped by run metadata.

    The legacy ``fit_type`` argument still selects a single fit. Passing
    ``fit_types`` enables multiple fits in one call.
    """

    normalized_fit_types = _normalize_fit_types(fit_type=fit_type, fit_types=fit_types)
    normalized_fit_ranges = _normalize_fit_ranges(fit_ranges)

    low_noise_value = DEFAULT_LOW_NOISE_MAX if low_noise_max is None else float(low_noise_max)
    if not np.isfinite(low_noise_value):
        raise ValueError("low_noise_max must be finite when provided")
    if low_noise_value < 0.0:
        raise ValueError("low_noise_max must be nonnegative")

    chance_value = float(chance_accuracy)
    if not np.isfinite(chance_value):
        raise ValueError("chance_accuracy must be finite")

    max_noise_value = None if max_noise_strength is None else float(max_noise_strength)
    if max_noise_value is not None and not np.isfinite(max_noise_value):
        raise ValueError("max_noise_strength must be finite when provided")

    selected_noise_strengths: tuple[float, ...] | None = None
    if noise_strengths is not None:
        selected_noise_strengths = tuple(float(value) for value in noise_strengths)
        if not selected_noise_strengths:
            raise ValueError("noise_strengths must not be empty when provided")
        if any(not np.isfinite(value) for value in selected_noise_strengths):
            raise ValueError("noise_strengths must be finite")

    min_points_by_fit_type = {
        current_fit_type: _required_points(current_fit_type, min_points)
        for current_fit_type in normalized_fit_types
    }

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        metric_value = row.get(metric_name)
        noise_strength = row.get("noise_strength")
        if metric_value is None or noise_strength is None:
            continue

        metadata = _group_metadata(row, metric_name=metric_name)
        key = tuple(_hashable_metadata_value(metadata[field]) for field in metadata)
        if key not in grouped:
            grouped[key] = {"metadata": metadata, "rows": []}
        grouped[key]["rows"].append(row)

    zne_rows: list[dict[str, Any]] = []
    for group in grouped.values():
        metadata = group["metadata"]
        total_points = _collect_points(
            group["rows"],
            metric_name=metric_name,
            max_noise_value=max_noise_value,
            selected_noise_strengths=selected_noise_strengths,
        )

        for current_fit_range in normalized_fit_ranges:
            range_points = _filter_points_for_range(
                total_points,
                fit_range=current_fit_range,
                low_noise_max=low_noise_value,
            )
            for current_fit_type in normalized_fit_types:
                zne_rows.append(
                    _fit_group_points(
                        metadata=metadata,
                        metric_name=metric_name,
                        fit_type=current_fit_type,
                        fit_range=current_fit_range,
                        total_points=total_points,
                        range_points=range_points,
                        low_noise_max=low_noise_value,
                        chance_accuracy=chance_value,
                        required_points=min_points_by_fit_type[current_fit_type],
                    )
                )

    zne_rows.sort(key=_zne_sort_key)
    return zne_rows


def summarize_zero_noise_extrapolation_directory(
    input_dir: str | Path,
    *,
    metric_name: str = "test_accuracy",
    fit_type: str | None = "linear",
    fit_types: Sequence[str] | None = None,
    fit_ranges: Sequence[str] | None = None,
    max_noise_strength: float | None = None,
    noise_strengths: Sequence[float] | None = None,
    low_noise_max: float | None = None,
    chance_accuracy: float = DEFAULT_CHANCE_ACCURACY,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    run_rows = _load_zne_input_rows(input_path, metric_name=metric_name)
    zne_rows = fit_zero_noise_extrapolation(
        run_rows,
        metric_name=metric_name,
        fit_type=fit_type,
        fit_types=fit_types,
        fit_ranges=fit_ranges,
        max_noise_strength=max_noise_strength,
        noise_strengths=noise_strengths,
        low_noise_max=low_noise_max,
        chance_accuracy=chance_accuracy,
    )

    resolved_output_json = input_path / "zne_summary.json" if output_json is None else Path(output_json)
    resolved_output_csv = input_path / "zne_summary.csv" if output_csv is None else Path(output_csv)
    resolved_output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)

    resolved_output_json.write_text(json.dumps(zne_rows, indent=2, sort_keys=True) + "\n")
    _write_zne_csv(resolved_output_csv, zne_rows)
    return {
        "zne_rows": zne_rows,
        "output_json": str(resolved_output_json.resolve()),
        "output_csv": str(resolved_output_csv.resolve()),
    }


def _normalize_fit_types(
    *,
    fit_type: str | None,
    fit_types: Sequence[str] | None,
) -> tuple[str, ...]:
    requested = (fit_type or "linear",) if fit_types is None else tuple(fit_types)
    if not requested:
        raise ValueError("fit_types must not be empty")

    normalized: list[str] = []
    for requested_fit_type in requested:
        value = str(requested_fit_type)
        if value not in SUPPORTED_FIT_TYPES:
            choices = "', '".join(SUPPORTED_FIT_TYPES)
            raise ValueError(f"fit_type must be one of '{choices}'")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _normalize_fit_ranges(fit_ranges: Sequence[str] | None) -> tuple[str, ...]:
    requested = ("full",) if fit_ranges is None else tuple(fit_ranges)
    if not requested:
        raise ValueError("fit_ranges must not be empty")

    normalized: list[str] = []
    for requested_fit_range in requested:
        value = str(requested_fit_range)
        if value not in SUPPORTED_FIT_RANGES:
            choices = "', '".join(SUPPORTED_FIT_RANGES)
            raise ValueError(f"fit_range must be one of '{choices}'")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _required_points(fit_type: str, min_points: int | None) -> int:
    default_min_points = 2 if fit_type == "log_margin_linear" else _POLYNOMIAL_FIT_DEGREES[fit_type] + 1
    if min_points is None:
        return default_min_points

    required_points = int(min_points)
    if required_points < default_min_points:
        raise ValueError(f"min_points must be at least {default_min_points} for fit_type='{fit_type}'")
    return required_points


def _load_zne_input_rows(input_path: Path, *, metric_name: str) -> list[dict[str, Any]]:
    try:
        run_rows = load_completed_noisy_comparison_runs(input_path)
    except ValueError:
        summary_path = input_path / "summary.csv"
        if summary_path.exists():
            return _load_csv_rows(summary_path)
        raise

    if any(row.get(metric_name) is not None for row in run_rows):
        return run_rows

    summary_path = input_path / "summary.csv"
    if summary_path.exists():
        summary_rows = _load_csv_rows(summary_path)
        if any(row.get(metric_name) is not None for row in summary_rows):
            return summary_rows
    return run_rows


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [
            {key: _parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _parse_csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return None

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _group_metadata(row: dict[str, Any], *, metric_name: str) -> dict[str, Any]:
    metadata = {field: row.get(field) for field in _GROUP_METADATA_FIELDS}
    if metadata["seed"] is None:
        metadata["seed"] = row.get("random_seed")
    metadata["mitigation_method"] = _mitigation_method(row, metric_name=metric_name)
    return metadata


def _mitigation_method(row: dict[str, Any], *, metric_name: str) -> str | None:
    explicit_method = row.get("mitigation_method")
    if explicit_method not in (None, ""):
        return str(explicit_method)

    methods: list[str] = []
    if bool(row.get("noise_aware_training", False)):
        methods.append("noise_aware_training")
    if bool(row.get("symmetry_regularization", False)):
        methods.append("symmetry_regularized_training")
    if metric_name.startswith("symmetry_twirled_"):
        methods.append("symmetry_twirled_evaluation")
    return "+".join(methods) if methods else None


def _collect_points(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    max_noise_value: float | None,
    selected_noise_strengths: Sequence[float] | None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        try:
            x_value = float(row["noise_strength"])
            y_value = float(row[metric_name])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        if max_noise_value is not None and x_value > max_noise_value:
            continue
        if selected_noise_strengths is not None and not _noise_strength_selected(
            x_value,
            selected_noise_strengths,
        ):
            continue
        points.append((x_value, y_value))

    points.sort(key=lambda point: point[0])
    return points


def _filter_points_for_range(
    points: list[tuple[float, float]],
    *,
    fit_range: str,
    low_noise_max: float,
) -> list[tuple[float, float]]:
    if fit_range == "full":
        return list(points)
    return [point for point in points if point[0] <= low_noise_max or np.isclose(point[0], low_noise_max)]


def _fit_group_points(
    *,
    metadata: dict[str, Any],
    metric_name: str,
    fit_type: str,
    fit_range: str,
    total_points: list[tuple[float, float]],
    range_points: list[tuple[float, float]],
    low_noise_max: float,
    chance_accuracy: float,
    required_points: int,
) -> dict[str, Any]:
    warnings_list: list[str] = []
    if fit_range == "full":
        warnings_list.append("full_range_fit_use_caution")

    if fit_type == "log_margin_linear":
        fit_points = [(x_value, y_value) for x_value, y_value in range_points if y_value > chance_accuracy]
        excluded_count = len(range_points) - len(fit_points)
        if excluded_count:
            warnings_list.append("log_margin_excluded_nonpositive_margin_points")
        result = _fit_log_margin_linear(
            fit_points,
            chance_accuracy=chance_accuracy,
            required_points=required_points,
        )
    else:
        fit_points = list(range_points)
        excluded_count = 0
        result = _fit_polynomial(
            fit_points,
            fit_type=fit_type,
            degree=_POLYNOMIAL_FIT_DEGREES[fit_type],
            required_points=required_points,
        )

    warnings_list.extend(result["warnings"])
    estimate = result["zero_noise_estimate"]
    if estimate is not None and (estimate < 0.0 or estimate > 1.0):
        warnings_list.append("estimate_out_of_bounds")

    warning_text = _join_warnings(warnings_list)
    noise_strengths_used = [float(point[0]) for point in fit_points]
    observed_metric_values_used = [float(point[1]) for point in fit_points]
    max_used_noise = float(np.max(noise_strengths_used)) if noise_strengths_used else None

    zne_row = {
        **metadata,
        "metric_name": metric_name,
        "fit_type": fit_type,
        "fit_range": fit_range,
        "low_noise_max": float(low_noise_max) if fit_range == "low_noise" else None,
        "chance_accuracy": float(chance_accuracy),
        "num_points_total": int(len(total_points)),
        "num_points_in_range": int(len(range_points)),
        "num_points_used": int(len(fit_points)),
        "num_points_excluded_by_log_margin": int(excluded_count),
        "noise_strengths_used": noise_strengths_used,
        "observed_metric_values_used": observed_metric_values_used,
        "zero_noise_estimate": estimate,
        "fit_valid": bool(result["fit_valid"]),
        "fit_warning": warning_text,
        "fit_coefficients": result["fit_coefficients"],
        "r_squared": result["r_squared"],
        "residual_mse": result["residual_mse"],
        "residual_mse_space": result["residual_mse_space"],
        "fit_target_space": result["fit_target_space"],
        "transformed_metric_values_used": result["transformed_metric_values_used"],
        "zne_metric_name": metric_name,
        "zne_fit_type": fit_type,
        "zne_fit_range": fit_range,
        "zne_estimate": estimate,
        "zne_intercept": estimate if fit_type != "log_margin_linear" else None,
        "zne_num_points": int(len(fit_points)),
        "zne_noise_strengths_used": noise_strengths_used,
        "zne_max_noise_strength": max_used_noise,
        "zne_residual_mse": result["residual_mse"],
        "zne_slope": result["aliases"].get("zne_slope"),
        "zne_quadratic_coeff": result["aliases"].get("zne_quadratic_coeff"),
        "zne_linear_coeff": result["aliases"].get("zne_linear_coeff"),
        "log_margin_intercept": result["aliases"].get("log_margin_intercept"),
        "log_margin_slope": result["aliases"].get("log_margin_slope"),
    }
    return zne_row


def _fit_polynomial(
    points: list[tuple[float, float]],
    *,
    fit_type: str,
    degree: int,
    required_points: int,
) -> dict[str, Any]:
    invalid_reason = _insufficient_points_warning(points, required_points=required_points, required_unique_x=degree + 1)
    if invalid_reason is not None:
        return _invalid_fit_result(
            warning=invalid_reason,
            fit_target_space="accuracy",
            residual_mse_space="accuracy",
        )

    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    rank_warning = _numpy_rank_warning_type()
    try:
        with warnings.catch_warnings():
            if rank_warning is not None:
                warnings.simplefilter("error", rank_warning)
            coefficients = np.polyfit(x, y, deg=degree)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError, Warning):
        return _invalid_fit_result(
            warning="fit_failed",
            fit_target_space="accuracy",
            residual_mse_space="accuracy",
        )

    fitted = np.polyval(coefficients, x)
    residual_mse = float(np.mean((y - fitted) ** 2))
    intercept = float(coefficients[-1])
    fit_coefficients: dict[str, float]
    aliases: dict[str, float] = {}
    if fit_type == "linear":
        slope = float(coefficients[0])
        fit_coefficients = {"slope": slope, "intercept": intercept}
        aliases["zne_slope"] = slope
    else:
        quadratic_coeff = float(coefficients[0])
        linear_coeff = float(coefficients[1])
        fit_coefficients = {
            "quadratic": quadratic_coeff,
            "linear": linear_coeff,
            "intercept": intercept,
        }
        aliases["zne_quadratic_coeff"] = quadratic_coeff
        aliases["zne_linear_coeff"] = linear_coeff
        aliases["zne_slope"] = linear_coeff

    return {
        "fit_valid": True,
        "warnings": [],
        "zero_noise_estimate": intercept,
        "fit_coefficients": fit_coefficients,
        "r_squared": _r_squared(y, fitted),
        "residual_mse": residual_mse,
        "residual_mse_space": "accuracy",
        "fit_target_space": "accuracy",
        "transformed_metric_values_used": None,
        "aliases": aliases,
    }


def _fit_log_margin_linear(
    points: list[tuple[float, float]],
    *,
    chance_accuracy: float,
    required_points: int,
) -> dict[str, Any]:
    invalid_reason = _insufficient_points_warning(points, required_points=required_points, required_unique_x=2)
    transformed_values = [float(np.log(point[1] - chance_accuracy)) for point in points]
    if invalid_reason is not None:
        warning = "insufficient_positive_margin_points" if len(points) < required_points else invalid_reason
        return _invalid_fit_result(
            warning=warning,
            fit_target_space="log_accuracy_margin",
            residual_mse_space="log_accuracy_margin",
            transformed_metric_values_used=transformed_values,
        )

    x = np.asarray([point[0] for point in points], dtype=np.float64)
    transformed_y = np.asarray(transformed_values, dtype=np.float64)
    rank_warning = _numpy_rank_warning_type()
    try:
        with warnings.catch_warnings():
            if rank_warning is not None:
                warnings.simplefilter("error", rank_warning)
            coefficients = np.polyfit(x, transformed_y, deg=1)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError, Warning):
        return _invalid_fit_result(
            warning="fit_failed",
            fit_target_space="log_accuracy_margin",
            residual_mse_space="log_accuracy_margin",
            transformed_metric_values_used=[float(value) for value in transformed_y],
        )

    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    zero_noise_estimate = float(chance_accuracy + np.exp(intercept))
    fitted = np.polyval(coefficients, x)
    residual_mse = float(np.mean((transformed_y - fitted) ** 2))

    return {
        "fit_valid": True,
        "warnings": [],
        "zero_noise_estimate": zero_noise_estimate,
        "fit_coefficients": {
            "log_margin_slope": slope,
            "log_margin_intercept": intercept,
            "transformed_space": "log_accuracy_margin",
        },
        "r_squared": _r_squared(transformed_y, fitted),
        "residual_mse": residual_mse,
        "residual_mse_space": "log_accuracy_margin",
        "fit_target_space": "log_accuracy_margin",
        "transformed_metric_values_used": transformed_values,
        "aliases": {
            "log_margin_slope": slope,
            "log_margin_intercept": intercept,
            "zne_slope": slope,
        },
    }


def _insufficient_points_warning(
    points: list[tuple[float, float]],
    *,
    required_points: int,
    required_unique_x: int,
) -> str | None:
    unique_x_values = {point[0] for point in points}
    if len(points) < required_points:
        return "insufficient_points"
    if len(unique_x_values) < required_unique_x:
        return "insufficient_unique_noise_strengths"
    return None


def _invalid_fit_result(
    *,
    warning: str,
    fit_target_space: str,
    residual_mse_space: str,
    transformed_metric_values_used: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "fit_valid": False,
        "warnings": [warning],
        "zero_noise_estimate": None,
        "fit_coefficients": None,
        "r_squared": None,
        "residual_mse": None,
        "residual_mse_space": residual_mse_space,
        "fit_target_space": fit_target_space,
        "transformed_metric_values_used": transformed_metric_values_used,
        "aliases": {},
    }


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float | None:
    residual_sum_squares = float(np.sum((y - fitted) ** 2))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    if np.isclose(total_sum_squares, 0.0):
        return 1.0 if np.isclose(residual_sum_squares, 0.0) else None
    return float(1.0 - residual_sum_squares / total_sum_squares)


def _join_warnings(warnings_list: Sequence[str]) -> str:
    return ";".join(dict.fromkeys(warnings_list))


def _write_zne_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fieldnames = _PREFERRED_CSV_FIELDS
    else:
        remaining = sorted({key for row in rows for key in row if key not in _PREFERRED_CSV_FIELDS})
        fieldnames = tuple(field for field in _PREFERRED_CSV_FIELDS if any(field in row for row in rows))
        fieldnames = fieldnames + tuple(remaining)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _noise_strength_selected(value: float, selected_noise_strengths: Sequence[float]) -> bool:
    return any(np.isclose(value, selected, rtol=1e-9, atol=1e-12) for selected in selected_noise_strengths)


def _hashable_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_metadata_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_hashable_metadata_value(item) for item in value)
    return value


def _sort_value(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    return (1, str(value))


def _zne_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _sort_value(row.get("backend_name")),
        _sort_value(row.get("model_family")),
        _sort_value(row.get("num_qubits")),
        _sort_value(row.get("train_size")),
        _sort_value(row.get("epochs")),
        _sort_value(row.get("noise_model_name")),
        _sort_value(row.get("mitigation_method")),
        _sort_value(row.get("noise_application_scope")),
        _sort_value(row.get("noisy_qubit_index")),
        _sort_value(row.get("coherent_overrotation_mode")),
        _sort_value(row.get("seed")),
        _sort_value(row.get("fit_range")),
        _sort_value(row.get("fit_type")),
    )


__all__ = [
    "fit_zero_noise_extrapolation",
    "summarize_zero_noise_extrapolation_directory",
]
