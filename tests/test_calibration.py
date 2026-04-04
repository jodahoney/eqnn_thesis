from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eqnn.backends import TORCH_AVAILABLE
from eqnn.cli import main as cli_main
from eqnn.experiments import (
    CalibrationSweepConfig,
    calibration_job_from_index,
    enumerate_calibration_jobs,
    run_calibration_sweep,
)


class CalibrationSweepTests(unittest.TestCase):
    def _tiny_config(self) -> CalibrationSweepConfig:
        return CalibrationSweepConfig(
            model_families=("su2_qcnn", "hea_qcnn"),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            dense_test_points=11,
        )

    def test_enumerate_jobs_and_indexing_are_consistent(self) -> None:
        config = CalibrationSweepConfig(
            model_families=("su2_qcnn", "hea_qcnn"),
            num_qubits_values=(4,),
            train_sizes=(2, 4),
            epochs_values=(1, 3),
            random_seeds=(0, 1),
            dense_test_points=11,
        )

        jobs = enumerate_calibration_jobs(config)

        self.assertEqual(len(jobs), 16)
        self.assertEqual(jobs[0], calibration_job_from_index(config, 0))
        self.assertEqual(jobs[-1], calibration_job_from_index(config, len(jobs) - 1))
        self.assertEqual(jobs[0].model_family, "su2_qcnn")
        self.assertEqual(jobs[-1].model_family, "hea_qcnn")

    def test_calibration_sweep_writes_summary_outputs(self) -> None:
        config = self._tiny_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "calibration"
            results = run_calibration_sweep(config, output_dir)

            self.assertEqual(len(results["runs"]), 2)
            self.assertEqual(len(results["summary"]), 2)
            self.assertTrue((output_dir / "runs.json").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())

            rows = list(csv.DictReader((output_dir / "summary.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["model_family"] for row in rows}, {"su2_qcnn", "hea_qcnn"})
            for row in rows:
                self.assertEqual(row["backend_name"], "numpy_pure")
                self.assertIn("mean_test_accuracy", row)
                self.assertIn("variance_test_loss", row)
                self.assertEqual(row["num_runs"], "1")

    def test_calibration_job_mode_then_aggregate_only(self) -> None:
        config = self._tiny_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "calibration"
            run_calibration_sweep(config, output_dir, job_index=0)
            run_calibration_sweep(config, output_dir, job_index=1)
            results = run_calibration_sweep(config, output_dir, aggregate_only=True)

            self.assertEqual(len(results["runs"]), 2)
            self.assertEqual(len(results["summary"]), 2)

    def test_cli_smoke_runs_calibration_for_su2_and_hea(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "cli_calibration"
            exit_code = cli_main(
                [
                    "run-calibration-sweep",
                    "--model-families",
                    "su2_qcnn",
                    "hea_qcnn",
                    "--num-qubits-values",
                    "4",
                    "--train-sizes",
                    "2",
                    "--epochs-values",
                    "1",
                    "--random-seeds",
                    "0",
                    "--dense-test-points",
                    "11",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "summary.csv").exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_torch_backend_calibration_job_writes_artifacts(self) -> None:
        config = CalibrationSweepConfig(
            model_families=("su2_qcnn",),
            backend_name="torch_pure",
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            dense_test_points=11,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "torch_calibration"
            result = run_calibration_sweep(config, output_dir, job_index=0)

            run = result["run"]
            run_output_dir = Path(run["output_dir"])

            self.assertEqual(run["backend_name"], "torch_pure")
            self.assertTrue(run_output_dir.exists())
            self.assertTrue((run_output_dir / "metrics.json").exists())
            self.assertTrue((run_output_dir / "best_parameters.npy").exists())
