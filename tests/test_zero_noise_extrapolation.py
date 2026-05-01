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
    def _row(self, noise_strength: float, test_accuracy: float) -> dict[str, object]:
        return {
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
            "test_accuracy": test_accuracy,
        }

    def test_linear_fit_extrapolates_synthetic_rows_to_zero_noise_limit(self) -> None:
        rows = [
            self._row(noise_strength, 1.0 - 2.0 * noise_strength)
            for noise_strength in (0.0, 0.05, 0.1)
        ]

        zne_rows = fit_zero_noise_extrapolation(rows, metric_name="test_accuracy", fit_type="linear")

        self.assertEqual(len(zne_rows), 1)
        self.assertAlmostEqual(zne_rows[0]["zne_estimate"], 1.0, places=6)
        self.assertAlmostEqual(zne_rows[0]["zne_slope"], -2.0, places=6)
        self.assertEqual(zne_rows[0]["zne_num_points"], 3)
        self.assertEqual(zne_rows[0]["zne_noise_strengths_used"], [0.0, 0.05, 0.1])
        self.assertAlmostEqual(zne_rows[0]["zne_residual_mse"], 0.0, places=12)

    def test_quadratic_fit_extrapolates_synthetic_rows_to_zero_noise_limit(self) -> None:
        rows = [
            self._row(noise_strength, 1.0 - 2.0 * noise_strength + 3.0 * noise_strength**2)
            for noise_strength in (0.0, 0.03, 0.06, 0.09)
        ]

        zne_rows = fit_zero_noise_extrapolation(rows, metric_name="test_accuracy", fit_type="quadratic")

        self.assertEqual(len(zne_rows), 1)
        self.assertAlmostEqual(zne_rows[0]["zne_estimate"], 1.0, places=6)
        self.assertAlmostEqual(zne_rows[0]["zne_quadratic_coeff"], 3.0, places=6)
        self.assertAlmostEqual(zne_rows[0]["zne_linear_coeff"], -2.0, places=6)
        self.assertEqual(zne_rows[0]["zne_fit_type"], "quadratic")

    def test_max_noise_strength_filter_excludes_high_noise_points(self) -> None:
        rows = [
            self._row(0.0, 1.0),
            self._row(0.05, 0.9),
            self._row(0.1, 0.2),
        ]

        zne_rows = fit_zero_noise_extrapolation(
            rows,
            metric_name="test_accuracy",
            fit_type="linear",
            max_noise_strength=0.05,
        )

        self.assertEqual(len(zne_rows), 1)
        self.assertAlmostEqual(zne_rows[0]["zne_estimate"], 1.0, places=6)
        self.assertEqual(zne_rows[0]["zne_noise_strengths_used"], [0.0, 0.05])
        self.assertEqual(zne_rows[0]["zne_max_noise_strength"], 0.05)

    def test_explicit_noise_strengths_filter_excludes_other_points(self) -> None:
        rows = [
            self._row(0.0, 1.0),
            self._row(0.01, 0.98),
            self._row(0.03, 0.94),
            self._row(0.05, 0.9),
            self._row(0.1, 0.2),
        ]

        zne_rows = fit_zero_noise_extrapolation(
            rows,
            metric_name="test_accuracy",
            fit_type="linear",
            noise_strengths=(0.0, 0.01, 0.03, 0.05),
        )

        self.assertEqual(len(zne_rows), 1)
        self.assertAlmostEqual(zne_rows[0]["zne_estimate"], 1.0, places=6)
        self.assertEqual(zne_rows[0]["zne_noise_strengths_used"], [0.0, 0.01, 0.03, 0.05])

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
            for index, noise_strength in enumerate((0.0, 0.05, 0.1)):
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
                    "--fit-type",
                    "quadratic",
                    "--max-noise-strength",
                    "0.1",
                    "--noise-strengths",
                    "0.0",
                    "0.05",
                    "0.1",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((input_dir / "zne_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
