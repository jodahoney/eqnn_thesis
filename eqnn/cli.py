"""Command-line interface for repository utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eqnn.datasets.cache import load_or_generate_cached_dataset
from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.datasets.io import load_dataset_bundle, save_dataset_bundle
from eqnn.experiments import (
    BackendBenchmarkConfig,
    BenchmarkSweepConfig,
    CalibrationSweepConfig,
    ExperimentConfig,
    PaperReproductionConfig,
    run_backend_benchmark,
    run_benchmark_sweep,
    run_calibration_sweep,
    run_paper_reproduction_suite,
    run_training_experiment,
    summarize_experiment_directory,
)
from eqnn.training import TrainingConfig
from eqnn.utils.timing import RuntimeProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eqnn",
        description="Utilities for building and testing the EQNN simulator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "generate-dataset",
        help="Generate a bond-alternating Heisenberg ground-state dataset.",
    )
    dataset_parser.add_argument("--num-qubits", type=int, required=True)
    dataset_parser.add_argument("--ratio-min", type=float, default=0.4)
    dataset_parser.add_argument("--ratio-max", type=float, default=1.6)
    dataset_parser.add_argument("--num-points", type=int, default=31)
    dataset_parser.add_argument("--train-fraction", type=float, default=0.8)
    dataset_parser.add_argument("--critical-ratio", type=float, default=1.0)
    dataset_parser.add_argument("--exclusion-window", type=float, default=0.05)
    dataset_parser.add_argument(
        "--boundary",
        choices=("open", "periodic"),
        default="open",
    )
    dataset_parser.add_argument(
        "--eigensolver",
        choices=("auto", "dense", "sparse"),
        default="auto",
    )
    dataset_parser.add_argument(
        "--labeling-strategy",
        choices=("ratio_threshold", "partial_reflection"),
        default="ratio_threshold",
    )
    dataset_parser.add_argument("--diagnostic-window", type=float, default=0.05)
    dataset_parser.add_argument("--partial-reflection-pairs", type=int, default=None)
    dataset_parser.add_argument("--split-seed", type=int, default=0)
    dataset_parser.add_argument("--output-dir", type=Path, required=True)
    dataset_parser.add_argument("--cache-dir", type=Path, default=None)
    dataset_parser.add_argument("--force-regenerate", action="store_true")
    dataset_parser.add_argument("--profile-json", type=Path, default=None)
    dataset_parser.set_defaults(handler=_handle_generate_dataset)

    experiment_parser = subparsers.add_parser(
        "run-experiment",
        help="Train a QCNN or baseline on a saved or freshly generated dataset.",
    )
    _add_dataset_loading_or_generation_args(experiment_parser)
    _add_model_args(experiment_parser)
    _add_training_args(experiment_parser)
    experiment_parser.add_argument("--output-dir", type=Path, required=True)
    experiment_parser.set_defaults(handler=_handle_run_experiment)

    sweep_parser = subparsers.add_parser(
        "run-benchmark-sweep",
        help="Run a small benchmark grid across datasets and model families.",
    )
    sweep_parser.add_argument("--num-qubits-values", type=int, nargs="+", required=True)
    sweep_parser.add_argument(
        "--model-families",
        nargs="+",
        choices=("su2_qcnn", "baseline_qcnn", "hea_qcnn"),
        default=("su2_qcnn", "baseline_qcnn"),
    )
    sweep_parser.add_argument(
        "--labeling-strategies",
        nargs="+",
        choices=("ratio_threshold", "partial_reflection"),
        default=("ratio_threshold",),
    )
    sweep_parser.add_argument(
        "--pooling-modes",
        nargs="+",
        choices=("partial_trace", "equivariant"),
        default=("partial_trace",),
    )
    sweep_parser.add_argument(
        "--readout-modes",
        nargs="+",
        choices=("swap", "dimerization"),
        default=("swap",),
    )
    sweep_parser.add_argument("--split-seeds", type=int, nargs="+", default=(0,))
    _add_dataset_generation_args(sweep_parser)
    _add_training_args(sweep_parser)
    sweep_parser.add_argument("--output-dir", type=Path, required=True)
    sweep_parser.set_defaults(handler=_handle_run_benchmark_sweep)

    backend_benchmark_parser = subparsers.add_parser(
        "benchmark-backends",
        help="Run the same experiment with multiple backends for runtime comparison.",
    )
    backend_benchmark_parser.add_argument(
        "--backends",
        nargs="+",
        choices=("numpy_pure", "torch_pure"),
        default=("numpy_pure", "torch_pure"),
    )
    backend_benchmark_parser.add_argument("--num-qubits", type=int, required=True)
    backend_benchmark_parser.add_argument(
        "--model-family",
        choices=("su2_qcnn", "baseline_qcnn", "hea_qcnn"),
        default="su2_qcnn",
    )
    backend_benchmark_parser.add_argument("--min-readout-qubits", type=int, default=None)
    backend_benchmark_parser.add_argument(
        "--pooling-mode",
        choices=("partial_trace", "equivariant"),
        default="partial_trace",
    )
    backend_benchmark_parser.add_argument(
        "--pooling-keep",
        choices=("left", "right"),
        default="left",
    )
    backend_benchmark_parser.add_argument(
        "--readout-mode",
        choices=("swap", "dimerization"),
        default="swap",
    )
    backend_benchmark_parser.add_argument(
        "--unshared-convolution-parameters",
        action="store_true",
        help="Use pair-specific convolution parameters instead of sharing within each parity sublayer.",
    )
    _add_dataset_generation_args(backend_benchmark_parser)
    _add_training_args(backend_benchmark_parser)
    backend_benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    backend_benchmark_parser.set_defaults(handler=_handle_benchmark_backends)

    calibration_parser = subparsers.add_parser(
        "run-calibration-sweep",
        help="Run a paper-style calibration sweep for choosing a practical epoch budget.",
    )
    calibration_parser.add_argument(
        "--model-families",
        nargs="+",
        choices=("su2_qcnn", "hea_qcnn", "baseline_qcnn"),
        default=("su2_qcnn", "hea_qcnn"),
    )
    calibration_parser.add_argument("--num-qubits-values", type=int, nargs="+", default=(6, 10))
    calibration_parser.add_argument("--train-sizes", type=int, nargs="+", default=(12,))
    calibration_parser.add_argument("--epochs-values", type=int, nargs="+", default=(50, 150, 300, 500, 750))
    calibration_parser.add_argument("--random-seeds", type=int, nargs="+", default=(0, 1, 2))
    calibration_parser.add_argument("--learning-rate", type=float, default=5e-2)
    calibration_parser.add_argument(
        "--gradient-backend",
        choices=("auto", "exact", "finite_difference"),
        default="exact",
    )
    calibration_parser.add_argument(
        "--initialization-strategy",
        choices=("current", "noisy_current"),
        default="noisy_current",
    )
    calibration_parser.add_argument("--initialization-noise-scale", type=float, default=5e-2)
    calibration_parser.add_argument("--critical-ratio", type=float, default=1.0)
    calibration_parser.add_argument("--left-ratio-min", type=float, default=0.0)
    calibration_parser.add_argument("--right-ratio-max", type=float, default=2.0)
    calibration_parser.add_argument("--dense-test-points", type=int, default=101)
    calibration_parser.add_argument(
        "--eigensolver",
        choices=("auto", "dense", "sparse"),
        default="auto",
    )
    calibration_parser.add_argument(
        "--backend-name",
        choices=("numpy_pure", "torch_pure"),
        default="numpy_pure",
    )
    calibration_parser.add_argument("--job-index", type=int, default=None)
    calibration_parser.add_argument("--aggregate-only", action="store_true")
    calibration_parser.add_argument("--force-rerun", action="store_true")
    calibration_parser.add_argument("--profile-json", type=Path, default=None)
    calibration_parser.add_argument("--output-dir", type=Path, required=True)
    calibration_parser.set_defaults(handler=_handle_run_calibration_sweep)

    reproduction_parser = subparsers.add_parser(
        "run-paper-reproduction",
        help="Run the locked paper_reproduction_v1 SU(2)-EQCNN baseline.",
    )
    reproduction_parser.add_argument("--num-qubits", type=int, required=True)
    reproduction_parser.add_argument("--train-sizes", type=int, nargs="+", default=(2, 4, 6, 8, 10, 12))
    reproduction_parser.add_argument("--random-seeds", type=int, nargs="+", default=(0, 1, 2))
    reproduction_parser.add_argument("--epochs", type=int, default=750)
    reproduction_parser.add_argument("--learning-rate", type=float, default=5e-2)
    reproduction_parser.add_argument(
        "--gradient-backend",
        choices=("auto", "exact", "finite_difference"),
        default="exact",
    )
    reproduction_parser.add_argument(
        "--initialization-strategy",
        choices=("current", "noisy_current"),
        default="noisy_current",
    )
    reproduction_parser.add_argument("--initialization-noise-scale", type=float, default=5e-2)
    reproduction_parser.add_argument("--critical-ratio", type=float, default=1.0)
    reproduction_parser.add_argument("--left-ratio-min", type=float, default=0.0)
    reproduction_parser.add_argument("--right-ratio-max", type=float, default=2.0)
    reproduction_parser.add_argument("--dense-test-points", type=int, default=101)
    reproduction_parser.add_argument(
        "--eigensolver",
        choices=("auto", "dense", "sparse"),
        default="auto",
    )
    reproduction_parser.add_argument("--cache-dir", type=Path, default=None)
    reproduction_parser.add_argument("--force-regenerate", action="store_true")
    reproduction_parser.add_argument("--profile-json", type=Path, default=None)
    reproduction_parser.add_argument("--output-dir", type=Path, required=True)
    reproduction_parser.set_defaults(handler=_handle_run_paper_reproduction)

    summary_parser = subparsers.add_parser(
        "summarize-experiments",
        help="Aggregate completed experiment directories into comparison tables.",
    )
    summary_parser.add_argument("--input-dir", type=Path, required=True)
    summary_parser.add_argument("--output-json", type=Path, default=None)
    summary_parser.add_argument("--output-csv", type=Path, default=None)
    summary_parser.add_argument("--num-qubits", type=int, default=None)
    summary_parser.add_argument(
        "--model-family",
        choices=("su2_qcnn", "baseline_qcnn", "hea_qcnn"),
        default=None,
    )
    summary_parser.add_argument(
        "--pooling-mode",
        choices=("partial_trace", "equivariant"),
        default=None,
    )
    summary_parser.add_argument(
        "--readout-mode",
        choices=("swap", "dimerization"),
        default=None,
    )
    summary_parser.set_defaults(handler=_handle_summarize_experiments)

    return parser


def _handle_generate_dataset(args: argparse.Namespace) -> int:
    config = _dataset_config_from_args(args)
    profile = RuntimeProfile()

    if args.cache_dir is not None:
        bundle, cache_path, cache_hit = load_or_generate_cached_dataset(
            config,
            cache_dir=args.cache_dir,
            profile=profile,
            force_rebuild=args.force_regenerate,
        )
        save_dataset_bundle(bundle, args.output_dir, profile=profile)
        print(
            f"{'Loaded' if cache_hit else 'Built'} cached dataset at "
            f"{cache_path.resolve()}"
        )
    else:
        bundle = generate_dataset(config, profile=profile)
        save_dataset_bundle(bundle, args.output_dir, profile=profile)

    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(profile.summary(), indent=2, sort_keys=True) + "\n")

    print(f"Wrote dataset to {args.output_dir.resolve()}")
    print(
        "Train samples: "
        f"{len(bundle.train)} | Test samples: {len(bundle.test)} | "
        f"State dimension: {bundle.train.states.shape[1]}"
    )
    return 0


def _handle_run_experiment(args: argparse.Namespace) -> int:
    profile = RuntimeProfile()
    dataset_bundle = _load_or_generate_dataset_from_args(args, profile=profile)

    experiment_config = ExperimentConfig(
        model_family=args.model_family,
        backend_name=args.backend,
        num_qubits=args.num_qubits,
        min_readout_qubits=args.min_readout_qubits,
        boundary=args.boundary,
        shared_convolution_parameter=not args.unshared_convolution_parameters,
        pooling_mode=args.pooling_mode,
        pooling_keep=args.pooling_keep,
        readout_mode=args.readout_mode,
    )
    training_config = _training_config_from_args(args)

    result = run_training_experiment(
        dataset_bundle,
        experiment_config,
        training_config,
        output_dir=args.output_dir,
        profile=profile,
    )

    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(profile.summary(), indent=2, sort_keys=True) + "\n")

    print(f"Experiment complete: {result['experiment_name']}")
    print(
        "Train accuracy/loss: "
        f"{result['train_metrics']['accuracy']:.3f} / {result['train_metrics']['loss']:.3f}"
    )
    print(
        "Test accuracy/loss: "
        f"{result['test_metrics']['accuracy']:.3f} / {result['test_metrics']['loss']:.3f}"
    )
    print(f"Artifacts written to {args.output_dir.resolve()}")
    return 0


def _handle_run_benchmark_sweep(args: argparse.Namespace) -> int:
    training_config = _training_config_from_args(args)
    sweep_config = BenchmarkSweepConfig(
        num_qubits_values=tuple(args.num_qubits_values),
        model_families=tuple(args.model_families),
        labeling_strategies=tuple(args.labeling_strategies),
        pooling_modes=tuple(args.pooling_modes),
        readout_modes=tuple(args.readout_modes),
        split_seeds=tuple(args.split_seeds),
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        num_points=args.num_points,
        train_fraction=args.train_fraction,
        critical_ratio=args.critical_ratio,
        exclusion_window=args.exclusion_window,
        diagnostic_window=args.diagnostic_window,
        boundary=args.boundary,
        eigensolver=args.eigensolver,
        partial_reflection_pairs=args.partial_reflection_pairs,
        training_config=training_config,
    )
    results = run_benchmark_sweep(sweep_config, args.output_dir)

    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps({"note": "benchmark sweep profiling not yet wired"}, indent=2) + "\n")

    print(f"Completed {len(results)} experiments")
    print(f"Summary written to {(args.output_dir / 'summary.csv').resolve()}")
    return 0


def _handle_benchmark_backends(args: argparse.Namespace) -> int:
    dataset_config = _dataset_config_from_args(args)
    training_config = _training_config_from_args(args)
    benchmark_config = BackendBenchmarkConfig(
        backend_names=tuple(args.backends),
        dataset_config=dataset_config,
        experiment_config=ExperimentConfig(
            model_family=args.model_family,
            backend_name=args.backends[0],
            num_qubits=args.num_qubits,
            min_readout_qubits=args.min_readout_qubits,
            boundary=args.boundary,
            shared_convolution_parameter=not args.unshared_convolution_parameters,
            pooling_mode=args.pooling_mode,
            pooling_keep=args.pooling_keep,
            readout_mode=args.readout_mode,
        ),
        training_config=training_config,
    )
    rows = run_backend_benchmark(benchmark_config, args.output_dir)

    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps({"note": "backend benchmark profiling is stored per run"}, indent=2) + "\n")

    print(f"Completed {len(rows)} backend comparison runs")
    print(f"Summary written to {(args.output_dir / 'summary.csv').resolve()}")
    return 0


def _handle_run_calibration_sweep(args: argparse.Namespace) -> int:
    profile = RuntimeProfile()
    config = CalibrationSweepConfig(
        model_families=tuple(args.model_families),
        num_qubits_values=tuple(args.num_qubits_values),
        train_sizes=tuple(args.train_sizes),
        epochs_values=tuple(args.epochs_values),
        random_seeds=tuple(args.random_seeds),
        learning_rate=args.learning_rate,
        gradient_backend=args.gradient_backend,
        initialization_strategy=args.initialization_strategy,
        initialization_noise_scale=args.initialization_noise_scale,
        critical_ratio=args.critical_ratio,
        left_ratio_min=args.left_ratio_min,
        right_ratio_max=args.right_ratio_max,
        dense_test_points=args.dense_test_points,
        eigensolver=args.eigensolver,
        backend_name=args.backend_name,
    )
    results = run_calibration_sweep(
        config,
        args.output_dir,
        job_index=args.job_index,
        aggregate_only=args.aggregate_only,
        force_rerun=args.force_rerun,
        profile=profile,
    )

    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(profile.summary(), indent=2, sort_keys=True) + "\n")

    if args.job_index is not None and not args.aggregate_only:
        print(f"Completed calibration job index {args.job_index}")
        print(f"Run artifacts written under {args.output_dir.resolve()}")
        return 0

    print(f"Aggregated {len(results['runs'])} calibration runs")
    print(f"Summary written to {(args.output_dir / 'summary.csv').resolve()}")
    return 0


def _handle_run_paper_reproduction(args: argparse.Namespace) -> int:
    profile = RuntimeProfile()

    config = PaperReproductionConfig(
        num_qubits=args.num_qubits,
        train_sizes=tuple(args.train_sizes),
        random_seeds=tuple(args.random_seeds),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_backend=args.gradient_backend,
        initialization_strategy=args.initialization_strategy,
        initialization_noise_scale=args.initialization_noise_scale,
        critical_ratio=args.critical_ratio,
        left_ratio_min=args.left_ratio_min,
        right_ratio_max=args.right_ratio_max,
        dense_test_points=args.dense_test_points,
        eigensolver=args.eigensolver,
    )

    results = run_paper_reproduction_suite(
        config,
        args.output_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_regenerate,
        profile=profile,
    )

    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(profile.summary(), indent=2, sort_keys=True) + "\n")

    print(f"Completed {len(results['runs'])} paper reproduction runs")
    print(f"Summary written to {(args.output_dir / 'summary.csv').resolve()}")
    return 0


def _handle_summarize_experiments(args: argparse.Namespace) -> int:
    filters = {
        name: value
        for name, value in {
            "num_qubits": args.num_qubits,
            "model_family": args.model_family,
            "pooling_mode": args.pooling_mode,
            "readout_mode": args.readout_mode,
        }.items()
        if value is not None
    }
    summary_rows = summarize_experiment_directory(
        args.input_dir,
        filters=filters,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )

    print(f"Aggregated {len(summary_rows)} comparison rows")
    for row in summary_rows:
        print(
            " | ".join(
                (
                    f"n={row['num_qubits']}",
                    f"family={row['model_family']}",
                    f"pooling={row['pooling_mode']}",
                    f"readout={row['readout_mode']}",
                    f"test_acc={row['mean_test_accuracy']:.3f}",
                    f"test_loss={row['mean_test_loss']:.3f}",
                    f"runs={row['num_runs']}",
                )
            )
        )

    if args.output_json is not None:
        print(f"JSON summary written to {args.output_json.resolve()}")
    if args.output_csv is not None:
        print(f"CSV summary written to {args.output_csv.resolve()}")
    return 0


def _add_dataset_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ratio-min", type=float, default=0.4)
    parser.add_argument("--ratio-max", type=float, default=1.6)
    parser.add_argument("--num-points", type=int, default=31)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--critical-ratio", type=float, default=1.0)
    parser.add_argument("--exclusion-window", type=float, default=0.05)
    parser.add_argument(
        "--boundary",
        choices=("open", "periodic"),
        default="open",
    )
    parser.add_argument(
        "--eigensolver",
        choices=("auto", "dense", "sparse"),
        default="auto",
    )
    parser.add_argument(
        "--labeling-strategy",
        choices=("ratio_threshold", "partial_reflection"),
        default="ratio_threshold",
    )
    parser.add_argument("--diagnostic-window", type=float, default=0.05)
    parser.add_argument("--partial-reflection-pairs", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--profile-json", type=Path, default=None)


def _add_dataset_loading_or_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--num-qubits", type=int, required=True)
    _add_dataset_generation_args(parser)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-family",
        choices=("su2_qcnn", "baseline_qcnn", "hea_qcnn"),
        default="su2_qcnn",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy_pure", "torch_pure"),
        default="numpy_pure",
    )
    parser.add_argument("--min-readout-qubits", type=int, default=None)
    parser.add_argument(
        "--pooling-mode",
        choices=("partial_trace", "equivariant"),
        default="partial_trace",
    )
    parser.add_argument(
        "--pooling-keep",
        choices=("left", "right"),
        default="left",
    )
    parser.add_argument(
        "--readout-mode",
        choices=("swap", "dimerization"),
        default="swap",
    )
    parser.add_argument(
        "--unshared-convolution-parameters",
        action="store_true",
        help="Use pair-specific convolution parameters instead of sharing within each parity sublayer.",
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--loss", choices=("bce", "mse"), default="bce")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--finite-difference-eps", type=float, default=1e-3)
    parser.add_argument(
        "--gradient-backend",
        choices=("auto", "exact", "finite_difference"),
        default="auto",
    )
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--restore-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--initialization-strategy",
        choices=("current", "noisy_current"),
        default="current",
    )
    parser.add_argument("--initialization-noise-scale", type=float, default=5e-2)
    parser.add_argument("--num-restarts", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-update",
        choices=("none", "paper_nearest_critical"),
        default="none",
    )
    parser.add_argument("--threshold-critical-ratio", type=float, default=1.0)


def _dataset_config_from_args(args: argparse.Namespace) -> HeisenbergDatasetConfig:
    return HeisenbergDatasetConfig(
        num_qubits=args.num_qubits,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        num_points=args.num_points,
        train_fraction=args.train_fraction,
        critical_ratio=args.critical_ratio,
        exclusion_window=args.exclusion_window,
        boundary=args.boundary,
        eigensolver=args.eigensolver,
        labeling_strategy=args.labeling_strategy,
        diagnostic_window=args.diagnostic_window,
        partial_reflection_pairs=args.partial_reflection_pairs,
        split_seed=args.split_seed,
    )


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss=args.loss,
        batch_size=args.batch_size,
        finite_difference_eps=args.finite_difference_eps,
        gradient_backend=args.gradient_backend,
        optimizer=args.optimizer,
        restore_best=args.restore_best,
        initialization_strategy=args.initialization_strategy,
        initialization_noise_scale=args.initialization_noise_scale,
        num_restarts=args.num_restarts,
        random_seed=args.random_seed,
        classification_threshold=args.classification_threshold,
        threshold_update=args.threshold_update,
        threshold_critical_ratio=args.threshold_critical_ratio,
    )


def _load_or_generate_dataset_from_args(
    args: argparse.Namespace,
    *,
    profile: RuntimeProfile | None = None,
) -> DatasetBundle:
    if args.dataset_dir is not None:
        return load_dataset_bundle(args.dataset_dir, profile=profile)

    config = _dataset_config_from_args(args)

    if args.cache_dir is not None:
        bundle, _, _ = load_or_generate_cached_dataset(
            config,
            cache_dir=args.cache_dir,
            profile=profile,
            force_rebuild=args.force_regenerate,
        )
        return bundle

    return generate_dataset(config, profile=profile)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
