from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eqnn.backends import QISKIT_AVAILABLE
from eqnn.cli import main as cli_main
from eqnn.experiments import (
    NoisyComparisonConfig,
    aggregate_noisy_comparison_runs,
    enumerate_noisy_comparison_jobs,
    noisy_comparison_job_from_index,
    run_noisy_comparison,
    summarize_noisy_comparison_directory,
)


class NoisyComparisonTests(unittest.TestCase):
    def _tiny_config(self) -> NoisyComparisonConfig:
        return NoisyComparisonConfig(
            model_families=("su2_qcnn", "hea_qcnn"),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.0, 0.01),
            dense_test_points=11,
        )

    def test_enumerate_jobs_and_indexing_are_consistent(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn", "hea_qcnn"),
            num_qubits_values=(4,),
            train_sizes=(2, 4),
            epochs_values=(1,),
            random_seeds=(0, 1),
            noise_strength_values=(0.0, 0.01),
            dense_test_points=11,
        )

        jobs = enumerate_noisy_comparison_jobs(config)

        self.assertEqual(len(jobs), 16)
        self.assertEqual(jobs[0], noisy_comparison_job_from_index(config, 0))
        self.assertEqual(jobs[-1], noisy_comparison_job_from_index(config, len(jobs) - 1))
        self.assertEqual(jobs[0].model_family, "su2_qcnn")
        self.assertEqual(jobs[-1].model_family, "hea_qcnn")

    def test_odd_qubits_only_filters_even_sizes_and_preserves_indexing(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4, 5, 6, 7),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.0,),
            odd_qubits_only=True,
            dense_test_points=11,
        )

        jobs = enumerate_noisy_comparison_jobs(config)

        self.assertEqual(config.resolved_num_qubits_values, (5, 7))
        self.assertEqual([job.num_qubits for job in jobs], [5, 7])
        self.assertEqual(jobs[1], noisy_comparison_job_from_index(config, 1))

    def test_config_validation_errors_are_clear(self) -> None:
        with self.assertRaisesRegex(ValueError, "No odd qubit counts remain"):
            NoisyComparisonConfig(
                model_families=("su2_qcnn",),
                num_qubits_values=(4, 6),
                train_sizes=(2,),
                epochs_values=(1,),
                random_seeds=(0,),
                noise_strength_values=(0.0,),
                odd_qubits_only=True,
                dense_test_points=11,
            )

        with self.assertRaisesRegex(ValueError, "must lie in \\[0, 1\\]"):
            NoisyComparisonConfig(
                model_families=("su2_qcnn",),
                num_qubits_values=(4,),
                train_sizes=(2,),
                epochs_values=(1,),
                random_seeds=(0,),
                noise_model_name="depolarizing",
                noise_strength_values=(1.5,),
                dense_test_points=11,
            )

    def test_aggregate_only_writes_summary_with_noise_fields(self) -> None:
        config = self._tiny_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "noisy"
            run_one = (
                output_dir
                / "qiskit_mixed"
                / "su2_qcnn"
                / "n4"
                / "train_size_2"
                / "epochs_1"
                / "noise_depolarizing_0"
                / "seed_0"
            )
            run_two = (
                output_dir
                / "qiskit_mixed"
                / "hea_qcnn"
                / "n4"
                / "train_size_2"
                / "epochs_1"
                / "noise_depolarizing_0p01"
                / "seed_0"
            )
            run_one.mkdir(parents=True, exist_ok=True)
            run_two.mkdir(parents=True, exist_ok=True)

            run_one_row = {
                "job_index": 0,
                "experiment_name": "noisy_one",
                "backend_name": "qiskit_mixed",
                "model_family": "su2_qcnn",
                "num_qubits": 4,
                "train_size": 2,
                "epochs": 1,
                "noise_model_name": "depolarizing",
                "noise_strength": 0.0,
                "seed": 0,
                "train_accuracy": 1.0,
                "test_accuracy": 1.0,
                "train_loss": 0.4,
                "test_loss": 0.5,
                "classification_threshold": 0.5,
                "runtime_seconds": 0.1,
                "output_dir": str(run_one.resolve()),
            }
            run_two_row = {
                "job_index": 1,
                "experiment_name": "noisy_two",
                "backend_name": "qiskit_mixed",
                "model_family": "hea_qcnn",
                "num_qubits": 4,
                "train_size": 2,
                "epochs": 1,
                "noise_model_name": "depolarizing",
                "noise_strength": 0.01,
                "seed": 0,
                "train_accuracy": 0.75,
                "test_accuracy": 0.5,
                "train_loss": 0.7,
                "test_loss": 0.9,
                "classification_threshold": 0.52,
                "runtime_seconds": 0.2,
                "output_dir": str(run_two.resolve()),
            }
            (run_one / "noisy_run.json").write_text(json.dumps(run_one_row, indent=2, sort_keys=True) + "\n")
            (run_two / "noisy_run.json").write_text(json.dumps(run_two_row, indent=2, sort_keys=True) + "\n")

            results = run_noisy_comparison(config, output_dir, aggregate_only=True)

            self.assertEqual(len(results["runs"]), 2)
            self.assertEqual(len(results["summary"]), 2)
            self.assertTrue((output_dir / "summary.csv").exists())

            rows = list(csv.DictReader((output_dir / "summary.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["backend_name"], "qiskit_mixed")
                self.assertEqual(row["noise_model_name"], "depolarizing")
                self.assertIn("noise_strength", row)
                self.assertIn("mean_test_accuracy", row)
                self.assertIn("mean_build_time_seconds", row)
                self.assertIn("mean_total_training_time_seconds", row)

            aggregated = aggregate_noisy_comparison_runs(results["runs"])
            self.assertEqual(len(aggregated), 2)
            self.assertEqual({row["noise_strength"] for row in aggregated}, {0.0, 0.01})

    def test_recursive_summary_utility_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "noisy_summary"
            run_dir = (
                output_dir
                / "qiskit_mixed"
                / "su2_qcnn"
                / "n5"
                / "train_size_4"
                / "epochs_10"
                / "noise_depolarizing_0p1"
                / "seed_0"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "noisy_run.json").write_text(
                json.dumps(
                    {
                        "job_index": 0,
                        "experiment_name": "recursive_summary",
                        "backend_name": "qiskit_mixed",
                        "model_family": "su2_qcnn",
                        "num_qubits": 5,
                        "train_size": 4,
                        "epochs": 10,
                        "noise_model_name": "depolarizing",
                        "noise_strength": 0.1,
                        "seed": 0,
                        "train_accuracy": 0.75,
                        "test_accuracy": 0.5,
                        "train_loss": 0.8,
                        "test_loss": 1.0,
                        "classification_threshold": 0.51,
                        "build_time_seconds": 0.02,
                        "forward_time_seconds": 0.08,
                        "gradient_time_seconds": 0.12,
                        "total_training_time_seconds": 0.5,
                        "runtime_seconds": 0.7,
                        "output_dir": str(run_dir.resolve()),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            result = summarize_noisy_comparison_directory(output_dir)

            self.assertEqual(len(result["runs"]), 1)
            self.assertEqual(len(result["summary"]), 1)
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.json").exists())

    def test_cli_aggregate_only_smoke_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "cli_noisy"
            run_dir = (
                output_dir
                / "qiskit_mixed"
                / "su2_qcnn"
                / "n4"
                / "train_size_2"
                / "epochs_1"
                / "noise_depolarizing_0"
                / "seed_0"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "noisy_run.json").write_text(
                json.dumps(
                    {
                        "job_index": 0,
                        "experiment_name": "cli_noisy",
                        "backend_name": "qiskit_mixed",
                        "model_family": "su2_qcnn",
                        "num_qubits": 4,
                        "train_size": 2,
                        "epochs": 1,
                        "noise_model_name": "depolarizing",
                        "noise_strength": 0.0,
                        "seed": 0,
                        "train_accuracy": 1.0,
                        "test_accuracy": 1.0,
                        "train_loss": 0.4,
                        "test_loss": 0.5,
                        "classification_threshold": 0.5,
                        "runtime_seconds": 0.1,
                        "output_dir": str(run_dir.resolve()),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            exit_code = cli_main(
                [
                    "run-noisy-comparison",
                    "--model-families",
                    "su2_qcnn",
                    "--num-qubits-values",
                    "4",
                    "--train-sizes",
                    "2",
                    "--epochs-values",
                    "1",
                    "--random-seeds",
                    "0",
                    "--noise-strength-values",
                    "0.0",
                    "--dense-test-points",
                    "11",
                    "--aggregate-only",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "summary.csv").exists())

    def test_cli_summarize_noisy_comparison_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "cli_summary"
            run_dir = (
                output_dir
                / "qiskit_mixed"
                / "hea_qcnn"
                / "n5"
                / "train_size_4"
                / "epochs_10"
                / "noise_coherent_overrotation_0p1"
                / "seed_1"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "noisy_run.json").write_text(
                json.dumps(
                    {
                        "job_index": 0,
                        "experiment_name": "cli_summary",
                        "backend_name": "qiskit_mixed",
                        "model_family": "hea_qcnn",
                        "num_qubits": 5,
                        "train_size": 4,
                        "epochs": 10,
                        "noise_model_name": "coherent_overrotation",
                        "noise_strength": 0.1,
                        "seed": 1,
                        "train_accuracy": 0.6,
                        "test_accuracy": 0.4,
                        "train_loss": 0.9,
                        "test_loss": 1.1,
                        "classification_threshold": 0.5,
                        "build_time_seconds": 0.02,
                        "forward_time_seconds": 0.08,
                        "gradient_time_seconds": 0.1,
                        "total_training_time_seconds": 0.5,
                        "runtime_seconds": 0.75,
                        "output_dir": str(run_dir.resolve()),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            exit_code = cli_main(
                [
                    "summarize-noisy-comparison",
                    "--input-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "summary.csv").exists())

    @unittest.skipUnless(QISKIT_AVAILABLE, "qiskit is not installed")
    def test_qiskit_mixed_runner_smoke_writes_artifacts(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.01,),
            dense_test_points=11,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "noisy_run"
            result = run_noisy_comparison(config, output_dir, job_index=0)

            run = result["run"]
            run_output_dir = Path(run["output_dir"])

            self.assertEqual(run["backend_name"], "qiskit_mixed")
            self.assertEqual(run["noise_model_name"], "depolarizing")
            self.assertTrue(np.isfinite(run["train_loss"]))
            self.assertTrue(np.isfinite(run["test_loss"]))
            self.assertTrue((run_output_dir / "metrics.json").exists())
            self.assertTrue((run_output_dir / "best_parameters.npy").exists())
            self.assertTrue((run_output_dir / "noisy_job_config.json").exists())
            self.assertTrue((run_output_dir / "runtime_profile.json").exists())
            self.assertTrue((run_output_dir / "runtime_breakdown.json").exists())
            self.assertTrue((run_output_dir / "noisy_run_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
