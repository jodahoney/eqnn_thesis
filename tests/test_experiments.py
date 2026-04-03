from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from eqnn.cli import main as cli_main
from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.experiments import (
    BackendBenchmarkConfig,
    BenchmarkSweepConfig,
    ExperimentConfig,
    run_backend_benchmark,
    run_benchmark_sweep,
    run_training_experiment,
    summarize_experiment_directory,
)
from eqnn.backends import TORCH_AVAILABLE
from eqnn.training import TrainingConfig


class ExperimentRunnerTests(unittest.TestCase):
    def test_run_training_experiment_writes_artifacts(self) -> None:
        dataset = generate_dataset(HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=1))
        experiment_config = ExperimentConfig(
            model_family="baseline_qcnn",
            num_qubits=4,
            min_readout_qubits=4,
            readout_mode="dimerization",
        )
        training_config = TrainingConfig(epochs=2, learning_rate=0.1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "experiment"
            result = run_training_experiment(
                dataset,
                experiment_config,
                training_config,
                output_dir=output_dir,
            )

            self.assertIn("train_metrics", result)
            self.assertIn("test_metrics", result)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "best_parameters.npy").exists())
            self.assertTrue((output_dir / "train_predictions.npz").exists())
            self.assertTrue((output_dir / "test_predictions.npz").exists())

            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(metrics["experiment_config"]["model_family"], "baseline_qcnn")
            self.assertIn("history", metrics)

    def test_run_benchmark_sweep_writes_summary_files(self) -> None:
        sweep_config = BenchmarkSweepConfig(
            num_qubits_values=(4,),
            model_families=("su2_qcnn", "baseline_qcnn"),
            labeling_strategies=("ratio_threshold",),
            pooling_modes=("partial_trace",),
            readout_modes=("dimerization",),
            split_seeds=(0,),
            num_points=5,
            training_config=TrainingConfig(epochs=1, learning_rate=0.05),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "sweep"
            results = run_benchmark_sweep(sweep_config, output_dir)

            self.assertEqual(len(results), 2)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())

            rows = list(csv.DictReader((output_dir / "summary.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["model_family"] for row in rows}, {"su2_qcnn", "baseline_qcnn"})

    def test_summarize_experiment_directory_aggregates_completed_runs(self) -> None:
        sweep_config = BenchmarkSweepConfig(
            num_qubits_values=(4,),
            model_families=("su2_qcnn", "baseline_qcnn"),
            labeling_strategies=("ratio_threshold",),
            pooling_modes=("partial_trace",),
            readout_modes=("dimerization",),
            split_seeds=(0, 1),
            num_points=5,
            training_config=TrainingConfig(epochs=1, learning_rate=0.05),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "sweep"
            run_benchmark_sweep(sweep_config, output_dir)

            summary_json = output_dir / "aggregated.json"
            summary_csv = output_dir / "aggregated.csv"
            rows = summarize_experiment_directory(
                output_dir / "experiments",
                output_json=summary_json,
                output_csv=summary_csv,
            )

            self.assertEqual(len(rows), 2)
            self.assertTrue(summary_json.exists())
            self.assertTrue(summary_csv.exists())
            self.assertEqual({row["model_family"] for row in rows}, {"su2_qcnn", "baseline_qcnn"})
            self.assertEqual({row["num_runs"] for row in rows}, {2})

    def test_run_backend_benchmark_writes_summary_files(self) -> None:
        benchmark_config = BackendBenchmarkConfig(
            backend_names=("numpy_pure",),
            dataset_config=HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=2),
            experiment_config=ExperimentConfig(
                model_family="su2_qcnn",
                num_qubits=4,
                backend_name="numpy_pure",
            ),
            training_config=TrainingConfig(epochs=1, learning_rate=0.05),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "backend_benchmark"
            rows = run_backend_benchmark(benchmark_config, output_dir)

            self.assertEqual(len(rows), 1)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertEqual(rows[0]["backend_name"], "numpy_pure")

    def test_cli_benchmark_backends_smoke_with_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "cli_backend_benchmark"
            exit_code = cli_main(
                [
                    "benchmark-backends",
                    "--backends",
                    "numpy_pure",
                    "--num-qubits",
                    "4",
                    "--model-family",
                    "su2_qcnn",
                    "--num-points",
                    "5",
                    "--epochs",
                    "1",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "summary.csv").exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_run_backend_benchmark_supports_numpy_and_torch(self) -> None:
        benchmark_config = BackendBenchmarkConfig(
            backend_names=("numpy_pure", "torch_pure"),
            dataset_config=HeisenbergDatasetConfig(num_qubits=4, num_points=5, split_seed=3),
            experiment_config=ExperimentConfig(
                model_family="hea_qcnn",
                num_qubits=4,
                backend_name="numpy_pure",
            ),
            training_config=TrainingConfig(epochs=1, learning_rate=0.05, gradient_backend="exact"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "backend_benchmark_torch"
            rows = run_backend_benchmark(benchmark_config, output_dir)

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["backend_name"] for row in rows}, {"numpy_pure", "torch_pure"})
