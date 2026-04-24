from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eqnn.cli import main as cli_main
from eqnn.experiments.zero_noise_extrapolation import (
    fit_zero_noise_extrapolation,
    summarize_zero_noise_extrapolation_directory,
)


class ZeroNoiseExtrapolationTests(unittest.TestCase):
    def test_linear_fit_extrapolates_synthetic_rows_to_zero_noise_limit(self) -> None:
        rows = [
            {
                "backend_name": "qiskit_mixed",
                "model_family": "su2_qcnn",
                "num_qubits": 4,
                "train_size": 4,
                "epochs": 10,
                "seed": 0,
                "noise_model_name": "depolarizing",
                "noise_application_scope": "active",
                "noisy_qubit_index": None,
                "coherent_overrotation_mode": None,
                "noise_strength": noise_strength,
                "test_accuracy": 1.0 - 2.0 * noise_strength,
            }
            for noise_strength in (0.0, 0.05, 0.1)
        ]

        zne_rows = fit_zero_noise_extrapolation(rows, metric_name="test_accuracy", fit_type="linear")

        self.assertEqual(len(zne_rows), 1)
        self.assertAlmostEqual(zne_rows[0]["zne_estimate"], 1.0, places=6)
        self.assertAlmostEqual(zne_rows[0]["zne_slope"], -2.0, places=6)
        self.assertEqual(zne_rows[0]["zne_num_points"], 3)

    def test_summary_directory_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / "zne_input"
            run_dir = (
                input_dir
                / "qiskit_mixed"
                / "su2_qcnn"
                / "n4"
                / "train_size_4"
                / "epochs_10"
                / "noise_depolarizing_0p05"
                / "scope_active"
                / "noisy_qubit_none"
                / "seed_0"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            for index, noise_strength in enumerate((0.0, 0.05)):
                current_dir = run_dir.parent.parent / f"noise_depolarizing_{str(noise_strength).replace('.', 'p')}" / "scope_active" / "noisy_qubit_none" / "seed_0"
                current_dir.mkdir(parents=True, exist_ok=True)
                (current_dir / "noisy_run.json").write_text(
                    (
                        "{"
                        f"\"job_index\": {index}, "
                        "\"backend_name\": \"qiskit_mixed\", "
                        "\"model_family\": \"su2_qcnn\", "
                        "\"num_qubits\": 4, "
                        "\"train_size\": 4, "
                        "\"epochs\": 10, "
                        "\"seed\": 0, "
                        "\"noise_model_name\": \"depolarizing\", "
                        "\"noise_application_scope\": \"active\", "
                        "\"noisy_qubit_index\": null, "
                        "\"coherent_overrotation_mode\": null, "
                        f"\"noise_strength\": {noise_strength}, "
                        f"\"test_accuracy\": {1.0 - 2.0 * noise_strength}"
                        "}\n"
                    )
                )

            result = summarize_zero_noise_extrapolation_directory(input_dir)

            self.assertEqual(len(result["zne_rows"]), 1)
            self.assertTrue((input_dir / "zne_summary.json").exists())
            self.assertTrue((input_dir / "zne_summary.csv").exists())

    def test_cli_summarize_zne_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / "cli_zne"
            for index, noise_strength in enumerate((0.0, 0.1)):
                run_dir = (
                    input_dir
                    / "qiskit_mixed"
                    / "hea_qcnn"
                    / "n4"
                    / "train_size_4"
                    / "epochs_10"
                    / f"noise_depolarizing_{str(noise_strength).replace('.', 'p')}"
                    / "scope_active"
                    / "noisy_qubit_none"
                    / "seed_1"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "noisy_run.json").write_text(
                    (
                        "{"
                        f"\"job_index\": {index}, "
                        "\"backend_name\": \"qiskit_mixed\", "
                        "\"model_family\": \"hea_qcnn\", "
                        "\"num_qubits\": 4, "
                        "\"train_size\": 4, "
                        "\"epochs\": 10, "
                        "\"seed\": 1, "
                        "\"noise_model_name\": \"depolarizing\", "
                        "\"noise_application_scope\": \"active\", "
                        "\"noisy_qubit_index\": null, "
                        "\"coherent_overrotation_mode\": null, "
                        f"\"noise_strength\": {noise_strength}, "
                        f"\"test_accuracy\": {0.8 - noise_strength}"
                        "}\n"
                    )
                )

            exit_code = cli_main(
                [
                    "summarize-zne",
                    "--input-dir",
                    str(input_dir),
                    "--metric",
                    "test_accuracy",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((input_dir / "zne_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
