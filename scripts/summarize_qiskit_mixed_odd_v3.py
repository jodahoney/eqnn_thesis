#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get(d: dict[str, Any], *path: str, default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_seed_dir(name: str) -> int | None:
    m = re.fullmatch(r"seed_(\d+)", name)
    return int(m.group(1)) if m else None


def parse_suffix_int(name: str, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name)
    return int(m.group(1)) if m else None


def parse_float_from_noise_dir(name: str) -> float | None:
    """
    Examples:
      noise_coherent_overrotation_0      -> 0.0
      noise_coherent_overrotation_0p005  -> 0.005
      noise_depolarizing_0p1             -> 0.1
    """
    m = re.search(r"_([0-9]+(?:p[0-9]+)?)$", name)
    if not m:
        return None
    token = m.group(1).replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def infer_noise_model_name(noise_dir_name: str) -> str | None:
    """
    Examples:
      noise_depolarizing_0p01 -> depolarizing
      noise_coherent_overrotation_0p005 -> coherent_overrotation
    """
    if not noise_dir_name.startswith("noise_"):
        return None
    parts = noise_dir_name.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[1:-1])


def last_or_none(x: Any) -> Any:
    if isinstance(x, list) and len(x) > 0:
        return x[-1]
    return None


def extract_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    history = metrics.get("history", {}) or {}
    train_metrics = metrics.get("train_metrics", {}) or {}
    test_metrics = metrics.get("test_metrics", {}) or {}

    return {
        "classification_threshold": metrics.get("classification_threshold"),
        "parameter_count": metrics.get("parameter_count"),

        "train_loss": train_metrics.get("loss"),
        "test_loss": test_metrics.get("loss"),
        "train_accuracy": train_metrics.get("accuracy"),
        "test_accuracy": test_metrics.get("accuracy"),

        "history_final_loss": last_or_none(history.get("loss")),
        "history_final_accuracy": last_or_none(history.get("accuracy")),
        "history_best_loss": history.get("best_loss"),
        "history_best_accuracy": history.get("best_accuracy"),
        "best_restart": history.get("best_restart"),
    }


def extract_runtime(runtime_breakdown: dict[str, Any], runtime_profile: dict[str, Any]) -> dict[str, Any]:
    def pick(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    return {
        "build_time_sec": pick(
            runtime_breakdown.get("build_time_sec"),
            runtime_breakdown.get("build_seconds"),
            runtime_profile.get("build_time_sec"),
            runtime_profile.get("build_seconds"),
        ),
        "forward_time_sec": pick(
            runtime_breakdown.get("forward_time_sec"),
            runtime_breakdown.get("forward_seconds"),
            runtime_profile.get("forward_time_sec"),
            runtime_profile.get("forward_seconds"),
        ),
        "gradient_time_sec": pick(
            runtime_breakdown.get("gradient_time_sec"),
            runtime_breakdown.get("gradient_seconds"),
            runtime_profile.get("gradient_time_sec"),
            runtime_profile.get("gradient_seconds"),
        ),
        "train_time_sec": pick(
            runtime_breakdown.get("train_time_sec"),
            runtime_breakdown.get("training_time_sec"),
            runtime_breakdown.get("training_seconds"),
            runtime_profile.get("train_time_sec"),
            runtime_profile.get("training_time_sec"),
            runtime_profile.get("training_seconds"),
            runtime_profile.get("total_training_time_sec"),
        ),
        "total_time_sec": pick(
            runtime_breakdown.get("total_time_sec"),
            runtime_breakdown.get("total_seconds"),
            runtime_profile.get("total_time_sec"),
            runtime_profile.get("total_seconds"),
        ),
    }


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for metrics_path in root.rglob("metrics.json"):
        seed_dir = metrics_path.parent
        seed = parse_seed_dir(seed_dir.name)
        if seed is None:
            continue

        try:
            noise_dir = seed_dir.parent
            epochs_dir = noise_dir.parent
            train_size_dir = epochs_dir.parent
            n_dir = train_size_dir.parent
            model_dir = n_dir.parent
            backend_dir = model_dir.parent
        except Exception:
            continue

        metrics = load_json(metrics_path)

        metadata_path = seed_dir / "noisy_run_metadata.json"
        noisy_run_path = seed_dir / "noisy_run.json"
        runtime_breakdown_path = seed_dir / "runtime_breakdown.json"
        runtime_profile_path = seed_dir / "runtime_profile.json"

        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        noisy_run = load_json(noisy_run_path) if noisy_run_path.exists() else {}
        runtime_breakdown = load_json(runtime_breakdown_path) if runtime_breakdown_path.exists() else {}
        runtime_profile = load_json(runtime_profile_path) if runtime_profile_path.exists() else {}

        experiment_config = metrics.get("experiment_config", {}) or {}
        dataset_metadata = metrics.get("dataset_metadata", {}) or {}

        row = {
            "job_root": str(root),
            "run_dir": str(seed_dir),

            "backend_name": experiment_config.get(
                "backend_name",
                metadata.get("backend_name", backend_dir.name),
            ),
            "model_family": experiment_config.get(
                "model_family",
                metadata.get("model_family", model_dir.name),
            ),
            "num_qubits": dataset_metadata.get(
                "num_qubits",
                metadata.get("num_qubits", parse_suffix_int(n_dir.name, "n")),
            ),
            "train_size": dataset_metadata.get(
                "train_size",
                metadata.get("train_size", parse_suffix_int(train_size_dir.name, "train_size_")),
            ),
            "epochs": metadata.get(
                "epochs",
                parse_suffix_int(epochs_dir.name, "epochs_"),
            ),
            "random_seed": metadata.get("random_seed", seed),
            "noise_model_name": metadata.get(
                "noise_model_name",
                infer_noise_model_name(noise_dir.name),
            ),
            "noise_strength": metadata.get(
                "noise_strength",
                parse_float_from_noise_dir(noise_dir.name),
            ),
            "status": noisy_run.get("status", "completed"),
            "metrics_mtime": metrics_path.stat().st_mtime,
            "metrics_json_raw": json.dumps(metrics),
        }

        row.update(extract_metrics(metrics))
        row.update(extract_runtime(runtime_breakdown, runtime_profile))
        rows.append(row)

    return rows


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_keys = [
        "backend_name",
        "model_family",
        "num_qubits",
        "train_size",
        "epochs",
        "random_seed",
        "noise_model_name",
        "noise_strength",
    ]

    score_cols = [
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "history_final_loss",
        "history_final_accuracy",
        "history_best_loss",
        "history_best_accuracy",
        "build_time_sec",
        "forward_time_sec",
        "gradient_time_sec",
        "train_time_sec",
        "total_time_sec",
    ]

    existing_score_cols = [c for c in score_cols if c in df.columns]
    df = df.copy()
    df["non_null_score"] = df[existing_score_cols].notna().sum(axis=1)

    # Prefer rows with more populated fields, then newer files.
    df = df.sort_values(
        by=["non_null_score", "metrics_mtime"],
        ascending=[False, False],
    )
    df = df.drop_duplicates(subset=dedupe_keys, keep="first").copy()
    return df


def aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "backend_name",
        "model_family",
        "num_qubits",
        "train_size",
        "epochs",
        "noise_model_name",
        "noise_strength",
    ]

    numeric_cols = [
        "parameter_count",
        "classification_threshold",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "history_final_loss",
        "history_final_accuracy",
        "history_best_loss",
        "history_best_accuracy",
        "build_time_sec",
        "forward_time_sec",
        "gradient_time_sec",
        "train_time_sec",
        "total_time_sec",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            seeds_present=("random_seed", "nunique"),
            seeds=("random_seed", lambda s: ",".join(str(int(x)) for x in sorted(pd.unique(s.dropna())))),

            parameter_count_mean=("parameter_count", "mean"),
            classification_threshold_mean=("classification_threshold", "mean"),

            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),

            train_accuracy_mean=("train_accuracy", "mean"),
            train_accuracy_std=("train_accuracy", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),

            history_final_loss_mean=("history_final_loss", "mean"),
            history_final_loss_std=("history_final_loss", "std"),
            history_final_accuracy_mean=("history_final_accuracy", "mean"),
            history_final_accuracy_std=("history_final_accuracy", "std"),

            history_best_loss_mean=("history_best_loss", "mean"),
            history_best_loss_std=("history_best_loss", "std"),
            history_best_accuracy_mean=("history_best_accuracy", "mean"),
            history_best_accuracy_std=("history_best_accuracy", "std"),

            build_time_sec_mean=("build_time_sec", "mean"),
            forward_time_sec_mean=("forward_time_sec", "mean"),
            gradient_time_sec_mean=("gradient_time_sec", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            total_time_sec_mean=("total_time_sec", "mean"),
        )
        .reset_index()
        .sort_values(
            by=["num_qubits", "noise_model_name", "model_family", "train_size", "noise_strength"]
        )
    )

    return grouped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize all available qiskit_mixed_odd results across multiple job roots, with deduplication."
    )
    ap.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="One or more root directories such as data/noisy_comparisons/qiskit_mixed_odd_21032705",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary CSVs will be written.",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for root_str in args.input_roots:
        root = Path(root_str).resolve()
        if not root.exists():
            print(f"WARNING: skipping missing root: {root}")
            continue
        rows.extend(collect_rows(root))

    if not rows:
        raise SystemExit("No metrics.json runs found under the provided roots.")

    per_seed_df = pd.DataFrame(rows)

    # Coerce numerics before dedupe scoring and output.
    for col in [
        "num_qubits",
        "train_size",
        "epochs",
        "random_seed",
        "noise_strength",
        "parameter_count",
        "classification_threshold",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "history_final_loss",
        "history_final_accuracy",
        "history_best_loss",
        "history_best_accuracy",
        "build_time_sec",
        "forward_time_sec",
        "gradient_time_sec",
        "train_time_sec",
        "total_time_sec",
        "metrics_mtime",
    ]:
        if col in per_seed_df.columns:
            per_seed_df[col] = pd.to_numeric(per_seed_df[col], errors="coerce")

    deduped_df = deduplicate_rows(per_seed_df).sort_values(
        by=[
            "num_qubits",
            "noise_model_name",
            "model_family",
            "train_size",
            "noise_strength",
            "random_seed",
        ]
    )

    aggregated_df = aggregate_rows(deduped_df)

    per_seed_path = output_dir / "qiskit_mixed_odd_per_seed_summary.csv"
    aggregated_path = output_dir / "qiskit_mixed_odd_aggregated_summary.csv"

    deduped_df.to_csv(per_seed_path, index=False)
    aggregated_df.to_csv(aggregated_path, index=False)

    print(f"Wrote per-seed summary to: {per_seed_path}")
    print(f"Wrote aggregated summary to: {aggregated_path}")
    print(f"Raw rows found: {len(per_seed_df)}")
    print(f"Unique per-seed runs kept after dedupe: {len(deduped_df)}")
    print(f"Grouped rows: {len(aggregated_df)}")


if __name__ == "__main__":
    main()