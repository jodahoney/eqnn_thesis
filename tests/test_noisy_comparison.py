from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eqnn.backends import QISKIT_AVAILABLE
from eqnn.cli import build_parser, main as cli_main
from eqnn.experiments import (
    AllNoiseMitigationComparisonConfig,
    NoisyComparisonConfig,
    aggregate_noisy_comparison_runs,
    count_all_noise_mitigation_jobs,
    enumerate_all_noise_mitigation_configs,
    enumerate_noisy_comparison_jobs,
    noisy_comparison_job_from_index,
    run_noisy_comparison,
    summarize_noisy_comparison_directory,
)
from eqnn.experiments.noisy_comparison import (
    _build_training_noise_control,
    _expected_symmetry_breaking_metadata,
    _job_output_dir,
    _mitigation_metadata,
    _symmetry_regularization_metadata,
    _training_noise_metadata,
)
from eqnn.noise import noise_config_from_strength
from eqnn.verification import evaluate_with_symmetry_twirling


class ConstantProbabilityModel:
    def __init__(self, probability: float = 0.75) -> None:
        self.config = SimpleNamespace(num_qubits=1)
        self.probability = float(probability)
        self.threshold = 0.5

    def predict(self, state: np.ndarray, parameters: np.ndarray | None = None) -> float:
        del state, parameters
        return self.probability

    def get_classification_threshold(self) -> float:
        return self.threshold


class MutableNoiseBackend:
    def __init__(self) -> None:
        self.noise_config = noise_config_from_strength("depolarizing", 0.0)


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

    def _minimal_run_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "job_index": 0,
            "experiment_name": "synthetic_noisy",
            "backend_name": "qiskit_mixed",
            "model_family": "su2_qcnn",
            "num_qubits": 4,
            "train_size": 2,
            "epochs": 1,
            "noise_model_name": "depolarizing",
            "noise_strength": 0.0,
            "eval_noise_strength": 0.0,
            "seed": 0,
            "train_accuracy": 1.0,
            "test_accuracy": 1.0,
            "train_loss": 0.4,
            "test_loss": 0.5,
            "classification_threshold": 0.5,
            "runtime_seconds": 0.1,
        }
        row.update(overrides)
        return row

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

    def test_coherent_overrotation_and_noisy_qubit_indices_expand_jobs(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(5,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_model_name="coherent_overrotation",
            noise_strength_values=(0.01,),
            coherent_overrotation_mode="stochastic",
            noisy_qubit_indices=(None, 0, 2),
            dense_test_points=11,
        )

        jobs = enumerate_noisy_comparison_jobs(config)

        self.assertEqual(len(jobs), 3)
        self.assertEqual([job.noisy_qubit_index for job in jobs], [None, 0, 2])
        self.assertEqual(config.noise_model_name, "coherent_overrotation")

    def test_all_noise_cli_help_and_multi_value_parsing(self) -> None:
        parser = build_parser()

        help_text = parser.format_help()
        args = parser.parse_args(
            [
                "run-all-noise-mitigation-comparison",
                "--noise-model-names",
                "depolarizing",
                "phase_damping",
                "--mitigation-methods",
                "none",
                "symmetry_regularized",
                "--noisy-qubit-indices",
                "none",
                "0",
                "3",
                "--output-root",
                "/tmp/all-noise",
            ]
        )

        self.assertIn("run-all-noise-mitigation-comparison", help_text)
        self.assertEqual(args.noise_model_names, ["depolarizing", "phase_damping"])
        self.assertEqual(args.mitigation_methods, ["none", "symmetry_regularized"])
        self.assertEqual(args.noisy_qubit_indices, [None, 0, 3])

    def test_all_noise_mitigation_expansion_counts_jobs_and_rejects_unknown_methods(self) -> None:
        config = AllNoiseMitigationComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(4,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_model_names=("depolarizing", "phase_damping"),
            noise_strength_values=(0.0, 0.01),
            mitigation_methods=("none", "symmetry_regularized"),
            symmetry_regularization_beta_values=(0.01, 0.1),
            dense_test_points=11,
        )

        specs = enumerate_all_noise_mitigation_configs(config)

        self.assertEqual(len(specs), 4)
        self.assertEqual(count_all_noise_mitigation_jobs(config), 12)
        self.assertEqual(
            {(spec.noise_model_name, spec.mitigation_method) for spec in specs},
            {
                ("depolarizing", "none"),
                ("depolarizing", "symmetry_regularized"),
                ("phase_damping", "none"),
                ("phase_damping", "symmetry_regularized"),
            },
        )
        symreg_spec = next(spec for spec in specs if spec.mitigation_method == "symmetry_regularized")
        self.assertIsNotNone(symreg_spec.config)
        self.assertEqual(
            [job.symmetry_regularization_beta for job in enumerate_noisy_comparison_jobs(symreg_spec.config)[:2]],
            [0.01, 0.1],
        )

        with self.assertRaisesRegex(ValueError, "mitigation_methods"):
            AllNoiseMitigationComparisonConfig(mitigation_methods=("unsupported",))

    def test_mitigation_and_expected_symmetry_breaking_metadata(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_model_name="phase_damping",
            noise_strength_values=(0.01,),
            mitigation_method="symmetry_regularized",
            noise_application_scope="active",
            noisy_qubit_indices=(0,),
            selected_noisy_qubit_pattern="single_edge",
            symmetry_regularization=True,
            symmetry_regularization_weight=0.01,
            dense_test_points=11,
        )
        job = enumerate_noisy_comparison_jobs(config)[0]
        resolved_noise = noise_config_from_strength(
            "phase_damping",
            0.01,
            noise_application_scope="selected_qubits",
            noisy_qubits=(0,),
        )

        metadata = _mitigation_metadata(config, resolved_noise, job)

        self.assertEqual(metadata["mitigation_method"], "symmetry_regularized")
        self.assertEqual(metadata["expected_symmetry_breaking"], "true")
        self.assertEqual(metadata["expected_symmetry_breaking_note"], "localized_noise_breaks_site_uniformity")
        self.assertEqual(metadata["selected_noisy_qubit_pattern"], "single_edge")
        self.assertEqual(
            _expected_symmetry_breaking_metadata("depolarizing", "active", None),
            ("unknown", "global_depolarizing_channel_effect_unclear"),
        )
        self.assertEqual(
            _expected_symmetry_breaking_metadata("amplitude_damping", "active", None),
            ("true", "nonunital_amplitude_damping"),
        )

    def test_default_job_output_dir_preserves_legacy_layout(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.01,),
            dense_test_points=11,
        )
        job = enumerate_noisy_comparison_jobs(config)[0]
        path = _job_output_dir(Path("/tmp/noisy"), config, job)

        self.assertNotIn("scope_active", str(path))
        self.assertNotIn("noisy_qubit_none", str(path))

    def test_noise_aware_job_output_dir_includes_train_noise_regime(self) -> None:
        config_with_zero = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.05,),
            dense_test_points=11,
            noise_aware_training=True,
            training_noise_strengths=(0.0, 0.01, 0.03, 0.05),
        )
        config_without_zero = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.05,),
            dense_test_points=11,
            noise_aware_training=True,
            training_noise_strengths=(0.01, 0.03, 0.05),
        )
        job_with_zero = enumerate_noisy_comparison_jobs(config_with_zero)[0]
        job_without_zero = enumerate_noisy_comparison_jobs(config_without_zero)[0]

        path_with_zero = _job_output_dir(Path("/tmp/noisy"), config_with_zero, job_with_zero)
        path_without_zero = _job_output_dir(Path("/tmp/noisy"), config_without_zero, job_without_zero)

        self.assertNotEqual(path_with_zero, path_without_zero)
        self.assertIn("train_noise_0_0p01_0p03_0p05", str(path_with_zero))
        self.assertIn("train_noise_0p01_0p03_0p05", str(path_without_zero))

    def test_symmetry_regularization_beta_values_expand_jobs_and_paths(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.0, 0.01),
            dense_test_points=11,
            symmetry_regularization=True,
            symmetry_regularization_beta_values=(0.0, 0.01, 0.1),
        )

        jobs = enumerate_noisy_comparison_jobs(config)
        paths = [_job_output_dir(Path("/tmp/noisy"), config, job) for job in jobs[:3]]

        self.assertEqual(len(jobs), 6)
        self.assertEqual([job.symmetry_regularization_beta for job in jobs[:3]], [0.0, 0.01, 0.1])
        self.assertEqual(config.resolved_symmetry_regularization_beta_values, (0.0, 0.01, 0.1))
        self.assertEqual(len(set(paths)), 3)
        self.assertIn("symreg_beta_0", str(paths[0]))
        self.assertIn("symreg_beta_0p01", str(paths[1]))
        self.assertIn("symreg_beta_0p1", str(paths[2]))

    def test_singular_symmetry_regularization_weight_preserves_legacy_beta_path(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.01,),
            dense_test_points=11,
            symmetry_regularization=True,
            symmetry_regularization_weight=0.01,
        )
        job = enumerate_noisy_comparison_jobs(config)[0]
        path = _job_output_dir(Path("/tmp/noisy"), config, job)

        self.assertEqual(config.resolved_symmetry_regularization_beta_values, (0.01,))
        self.assertEqual(job.symmetry_regularization_beta, 0.01)
        self.assertIn("mitigation_symmetry_regularized", str(path))
        self.assertIn("beta_0p01", str(path))
        self.assertNotIn("symreg_beta_0p01", str(path))

    def test_symmetry_regularization_metadata_records_beta_and_weight(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.01,),
            dense_test_points=11,
            symmetry_regularization=True,
            symmetry_regularization_beta_values=(0.0, 0.1),
        )

        zero_beta_metadata = _symmetry_regularization_metadata(
            config,
            {
                "history": {
                    "symmetry_penalty": [0.03, 0.02],
                    "symmetry_regularization_note": "configured_with_zero_weight",
                }
            },
            beta=0.0,
        )
        positive_beta_metadata = _symmetry_regularization_metadata(
            config,
            {"history": {"symmetry_penalty": [0.03, 0.01]}},
            beta=0.1,
        )

        self.assertEqual(zero_beta_metadata["symmetry_regularization_beta"], 0.0)
        self.assertEqual(zero_beta_metadata["symmetry_regularization_weight"], 0.0)
        self.assertFalse(zero_beta_metadata["symmetry_regularization_enabled"])
        self.assertEqual(
            zero_beta_metadata["symmetry_regularization_note"],
            "beta_zero_regularization_disabled",
        )
        self.assertEqual(zero_beta_metadata["final_symmetry_penalty"], 0.02)
        self.assertEqual(positive_beta_metadata["symmetry_regularization_beta"], 0.1)
        self.assertEqual(positive_beta_metadata["symmetry_regularization_weight"], 0.1)
        self.assertTrue(positive_beta_metadata["symmetry_regularization_enabled"])
        self.assertEqual(
            positive_beta_metadata["symmetry_regularization_note"],
            "finite_difference_objective_regularizer",
        )

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

        with self.assertRaisesRegex(ValueError, "requires at least one explicit noisy_qubit_index"):
            NoisyComparisonConfig(
                model_families=("su2_qcnn",),
                num_qubits_values=(5,),
                train_sizes=(2,),
                epochs_values=(1,),
                random_seeds=(0,),
                noise_strength_values=(0.0,),
                noise_application_scope="selected_qubits",
                noisy_qubit_indices=(None,),
                dense_test_points=11,
            )

        with self.assertRaisesRegex(ValueError, "num_symmetry_twirl_samples"):
            NoisyComparisonConfig(
                model_families=("su2_qcnn",),
                num_qubits_values=(4,),
                train_sizes=(2,),
                epochs_values=(1,),
                random_seeds=(0,),
                noise_strength_values=(0.0,),
                num_symmetry_twirl_samples=0,
                dense_test_points=11,
            )

        default_noise_aware_config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(1,),
            random_seeds=(0,),
            noise_strength_values=(0.0, 0.03),
            noise_aware_training=True,
            dense_test_points=11,
        )
        self.assertEqual(default_noise_aware_config.training_noise_strengths, (0.0, 0.03))
        self.assertTrue(default_noise_aware_config.training_noise_defaulted_from_evaluation_grid)

        with self.assertRaisesRegex(ValueError, "training_noise_strengths"):
            NoisyComparisonConfig(
                model_families=("su2_qcnn",),
                num_qubits_values=(4,),
                train_sizes=(2,),
                epochs_values=(1,),
                random_seeds=(0,),
                noise_strength_values=(0.0,),
                training_noise_strengths=(-0.01,),
                dense_test_points=11,
            )

        with self.assertRaisesRegex(ValueError, "symmetry_regularization_weight"):
            NoisyComparisonConfig(
                model_families=("su2_qcnn",),
                num_qubits_values=(4,),
                train_sizes=(2,),
                epochs_values=(1,),
                random_seeds=(0,),
                noise_strength_values=(0.0,),
                symmetry_regularization=True,
                symmetry_regularization_weight=-0.1,
                dense_test_points=11,
            )

        with self.assertRaisesRegex(ValueError, "symmetry_regularization_beta_values"):
            NoisyComparisonConfig(
                model_families=("su2_qcnn",),
                num_qubits_values=(4,),
                train_sizes=(2,),
                epochs_values=(1,),
                random_seeds=(0,),
                noise_strength_values=(0.0,),
                symmetry_regularization=True,
                symmetry_regularization_beta_values=(0.0, -0.01),
                dense_test_points=11,
            )

    def test_symmetry_twirled_evaluation_smoke_and_unavailable_metadata(self) -> None:
        states = np.asarray(
            [
                [1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j],
            ],
            dtype=np.complex128,
        )
        labels = np.asarray([1, 1], dtype=np.int64)

        result = evaluate_with_symmetry_twirling(
            ConstantProbabilityModel(0.75),
            states,
            parameters=np.asarray([0.0]),
            labels=labels,
            threshold=0.5,
            num_symmetry_samples=2,
            seed=123,
        )

        self.assertTrue(result["symmetry_twirling_available"])
        self.assertEqual(result["num_state_samples"], 2)
        self.assertEqual(result["num_symmetry_samples"], 2)
        np.testing.assert_allclose(result["raw_probabilities"], [0.75, 0.75])
        np.testing.assert_allclose(result["twirled_probabilities"], [0.75, 0.75])
        self.assertAlmostEqual(result["twirled_accuracy"], 1.0)
        self.assertAlmostEqual(
            result["symmetry_twirled_raw_subset_accuracy"],
            result["symmetry_twirled_test_accuracy"],
        )
        self.assertEqual(result["symmetry_twirled_subset_size"], 2)
        self.assertEqual(result["symmetry_twirled_num_correct_raw_subset"], 2)
        self.assertEqual(result["symmetry_twirled_num_correct_twirled_subset"], 2)
        self.assertAlmostEqual(result["mean_abs_twirling_shift"], 0.0)

        unavailable = evaluate_with_symmetry_twirling(
            object(),
            states,
            labels=labels,
            num_symmetry_samples=2,
        )

        self.assertFalse(unavailable["symmetry_twirling_available"])
        self.assertIn("predict", unavailable["symmetry_twirling_note"])
        self.assertEqual(unavailable["symmetry_twirled_subset_size"], 0)

    def test_noise_aware_training_control_updates_and_restores_noise_config(self) -> None:
        config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(3,),
            random_seeds=(0,),
            noise_strength_values=(0.08,),
            noise_aware_training=True,
            training_noise_strengths=(0.0, 0.01, 0.03),
            training_noise_seed=5,
            dense_test_points=11,
        )
        job = enumerate_noisy_comparison_jobs(config)[0]
        backend = MutableNoiseBackend()
        evaluation_noise_config = noise_config_from_strength("depolarizing", 0.08)

        control = _build_training_noise_control(config, job, evaluation_noise_config, backend)

        self.assertEqual(len(control["schedule"]), 3)
        self.assertEqual(sum(control["counts"].values()), 3)
        metadata = control["epoch_callback"](1, object())
        self.assertEqual(metadata["epoch"], 1)
        self.assertEqual(backend.noise_config.noise_model_name, "depolarizing")
        self.assertAlmostEqual(backend.noise_config.primary_strength, control["schedule"][0])
        control["restore_callback"](object())
        self.assertAlmostEqual(backend.noise_config.primary_strength, 0.08)

    def test_training_noise_metadata_records_zero_inclusion_and_eval_default(self) -> None:
        config_with_zero = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(2,),
            random_seeds=(3,),
            noise_strength_values=(0.0, 0.03),
            noise_aware_training=True,
            training_noise_strengths=(0.0, 0.01),
            dense_test_points=11,
        )
        job_with_zero = enumerate_noisy_comparison_jobs(config_with_zero)[0]
        backend = MutableNoiseBackend()
        control_with_zero = _build_training_noise_control(
            config_with_zero,
            job_with_zero,
            noise_config_from_strength("depolarizing", job_with_zero.noise_strength),
            backend,
        )
        metadata_with_zero = _training_noise_metadata(config_with_zero, {"history": {}}, control_with_zero)

        self.assertEqual(metadata_with_zero["train_noise_strength_values"], [0.0, 0.01])
        self.assertEqual(metadata_with_zero["train_noise_sampling_mode"], "per_epoch_random_choice")
        self.assertTrue(metadata_with_zero["train_noise_includes_zero"])
        self.assertFalse(metadata_with_zero["training_noise_defaulted_from_evaluation_grid"])
        self.assertEqual(metadata_with_zero["training_noise_effective_seed"], 3)

        defaulted_config = NoisyComparisonConfig(
            model_families=("su2_qcnn",),
            num_qubits_values=(4,),
            train_sizes=(2,),
            epochs_values=(2,),
            random_seeds=(0,),
            noise_strength_values=(0.01, 0.03),
            noise_aware_training=True,
            dense_test_points=11,
        )
        defaulted_job = enumerate_noisy_comparison_jobs(defaulted_config)[0]
        defaulted_control = _build_training_noise_control(
            defaulted_config,
            defaulted_job,
            noise_config_from_strength("depolarizing", defaulted_job.noise_strength),
            MutableNoiseBackend(),
        )
        defaulted_metadata = _training_noise_metadata(defaulted_config, {"history": {}}, defaulted_control)

        self.assertEqual(defaulted_metadata["train_noise_strength_values"], [0.01, 0.03])
        self.assertFalse(defaulted_metadata["train_noise_includes_zero"])
        self.assertTrue(defaulted_metadata["training_noise_defaulted_from_evaluation_grid"])
        self.assertIn("defaulted_from_eval_noise_grid", defaulted_metadata["training_noise_note"])

    def test_aggregation_keeps_training_noise_regimes_separate(self) -> None:
        base_row = {
            "job_index": 0,
            "backend_name": "qiskit_mixed",
            "model_family": "su2_qcnn",
            "num_qubits": 7,
            "train_size": 16,
            "epochs": 20,
            "noise_model_name": "depolarizing",
            "noise_strength": 0.05,
            "eval_noise_strength": 0.05,
            "noise_aware_training": True,
            "training_noise_sampling": "per_epoch",
            "train_noise_sampling_mode": "per_epoch_random_choice",
            "training_noise_seed": 11,
            "seed": 0,
            "train_accuracy": 0.8,
            "test_accuracy": 0.7,
            "train_loss": 0.4,
            "test_loss": 0.5,
            "classification_threshold": 0.5,
            "runtime_seconds": 1.0,
        }
        rows = [
            {
                **base_row,
                "training_noise_strengths": [0.0, 0.01, 0.03, 0.05],
                "train_noise_strength_values": [0.0, 0.01, 0.03, 0.05],
                "train_noise_includes_zero": True,
            },
            {
                **base_row,
                "job_index": 1,
                "training_noise_strengths": [0.01, 0.03, 0.05],
                "train_noise_strength_values": [0.01, 0.03, 0.05],
                "train_noise_includes_zero": False,
            },
        ]

        aggregated = aggregate_noisy_comparison_runs(rows)

        self.assertEqual(len(aggregated), 2)
        regimes = {
            (tuple(row["train_noise_strength_values"]), row["train_noise_includes_zero"])
            for row in aggregated
        }
        self.assertEqual(
            regimes,
            {
                ((0.0, 0.01, 0.03, 0.05), True),
                ((0.01, 0.03, 0.05), False),
            },
        )
        self.assertTrue(all(row["eval_noise_strength"] == 0.05 for row in aggregated))

    def test_aggregation_keeps_symmetry_beta_values_separate_with_train_noise(self) -> None:
        base_row = self._minimal_run_row(
            noise_strength=0.05,
            eval_noise_strength=0.05,
            noise_aware_training=True,
            training_noise_strengths=[0.01, 0.03, 0.05],
            train_noise_strength_values=[0.01, 0.03, 0.05],
            training_noise_sampling="per_epoch",
            train_noise_sampling_mode="per_epoch_random_choice",
            train_noise_includes_zero=False,
            training_noise_seed=17,
            symmetry_regularization=True,
            num_symmetry_regularization_samples=2,
            symmetry_regularization_frequency=1,
            symmetry_regularization_state_samples=1,
            symmetry_regularization_seed=23,
        )
        rows = [
            {
                **base_row,
                "job_index": 0,
                "symmetry_regularization_enabled": False,
                "symmetry_regularization_beta": 0.0,
                "symmetry_regularization_weight": 0.0,
                "symmetry_regularization_note": "beta_zero_regularization_disabled",
                "final_symmetry_penalty": 0.02,
            },
            {
                **base_row,
                "job_index": 1,
                "symmetry_regularization_enabled": True,
                "symmetry_regularization_beta": 0.01,
                "symmetry_regularization_weight": 0.01,
                "symmetry_regularization_note": "finite_difference_objective_regularizer",
                "final_symmetry_penalty": 0.015,
            },
            {
                **base_row,
                "job_index": 2,
                "symmetry_regularization_enabled": True,
                "symmetry_regularization_beta": 0.1,
                "symmetry_regularization_weight": 0.1,
                "symmetry_regularization_note": "finite_difference_objective_regularizer",
                "final_symmetry_penalty": 0.01,
            },
        ]

        aggregated = aggregate_noisy_comparison_runs(rows)

        self.assertEqual(len(aggregated), 3)
        self.assertEqual({row["symmetry_regularization_beta"] for row in aggregated}, {0.0, 0.01, 0.1})
        self.assertEqual({row["symmetry_regularization_weight"] for row in aggregated}, {0.0, 0.01, 0.1})
        self.assertEqual(
            {
                (row["symmetry_regularization_beta"], row["symmetry_regularization_enabled"])
                for row in aggregated
            },
            {(0.0, False), (0.01, True), (0.1, True)},
        )
        self.assertTrue(all(row["train_noise_strength_values"] == [0.01, 0.03, 0.05] for row in aggregated))
        self.assertTrue(all(row["train_noise_includes_zero"] is False for row in aggregated))
        self.assertTrue(all(row["eval_noise_strength"] == 0.05 for row in aggregated))
        self.assertTrue(all("final_symmetry_penalty" in row for row in aggregated))

    def test_aggregation_keeps_mitigation_method_noise_model_and_localization_separate(self) -> None:
        rows = [
            self._minimal_run_row(
                job_index=0,
                mitigation_method="none",
                noise_model_name="depolarizing",
                expected_symmetry_breaking="unknown",
                expected_symmetry_breaking_note="global_depolarizing_channel_effect_unclear",
                selected_noisy_qubit_pattern="none",
            ),
            self._minimal_run_row(
                job_index=1,
                mitigation_method="symmetry_regularized",
                noise_model_name="depolarizing",
                expected_symmetry_breaking="unknown",
                expected_symmetry_breaking_note="global_depolarizing_channel_effect_unclear",
                symmetry_regularization=True,
                symmetry_regularization_enabled=True,
                symmetry_regularization_beta=0.1,
                symmetry_regularization_weight=0.1,
                selected_noisy_qubit_pattern="none",
            ),
            self._minimal_run_row(
                job_index=2,
                mitigation_method="none",
                noise_model_name="phase_damping",
                expected_symmetry_breaking="true",
                expected_symmetry_breaking_note="basis_selective_dephasing",
                selected_noisy_qubit_pattern="none",
            ),
            self._minimal_run_row(
                job_index=3,
                mitigation_method="none",
                noise_model_name="phase_damping",
                expected_symmetry_breaking="true",
                expected_symmetry_breaking_note="localized_noise_breaks_site_uniformity",
                noise_application_scope="selected_qubits",
                noisy_qubit_index=0,
                noisy_qubits=[0],
                selected_noisy_qubit_pattern="single_qubit",
            ),
        ]

        aggregated = aggregate_noisy_comparison_runs(rows)

        self.assertEqual(len(aggregated), 4)
        self.assertEqual(
            {
                (row["noise_model_name"], row["mitigation_method"], row["selected_noisy_qubit_pattern"])
                for row in aggregated
            },
            {
                ("depolarizing", "none", "none"),
                ("depolarizing", "symmetry_regularized", "none"),
                ("phase_damping", "none", "none"),
                ("phase_damping", "none", "single_qubit"),
            },
        )
        symreg_row = next(row for row in aggregated if row["mitigation_method"] == "symmetry_regularized")
        self.assertEqual(symreg_row["symmetry_regularization_beta"], 0.1)
        localized_row = next(row for row in aggregated if row["selected_noisy_qubit_pattern"] == "single_qubit")
        self.assertEqual(localized_row["expected_symmetry_breaking"], "true")

    def test_all_noise_aggregation_preserves_stage2_and_stage3_fields(self) -> None:
        rows = [
            self._minimal_run_row(
                job_index=0,
                mitigation_method="noise_aware_training",
                noise_aware_training=True,
                training_noise_strengths=[0.01, 0.03, 0.05],
                train_noise_strength_values=[0.01, 0.03, 0.05],
                train_noise_sampling_mode="per_epoch_random_choice",
                train_noise_includes_zero=False,
            ),
            self._minimal_run_row(
                job_index=1,
                mitigation_method="symmetry_regularized",
                symmetry_regularization=True,
                symmetry_regularization_enabled=True,
                symmetry_regularization_beta=0.05,
                symmetry_regularization_weight=0.05,
                final_symmetry_penalty=0.02,
                final_equivariance_error_mean=0.03,
                final_equivariance_error_max=0.04,
            ),
        ]

        aggregated = aggregate_noisy_comparison_runs(rows)

        noise_aware_row = next(row for row in aggregated if row["mitigation_method"] == "noise_aware_training")
        symreg_row = next(row for row in aggregated if row["mitigation_method"] == "symmetry_regularized")
        self.assertEqual(noise_aware_row["train_noise_strength_values"], [0.01, 0.03, 0.05])
        self.assertFalse(noise_aware_row["train_noise_includes_zero"])
        self.assertEqual(symreg_row["symmetry_regularization_beta"], 0.05)
        self.assertEqual(symreg_row["symmetry_regularization_weight"], 0.05)
        self.assertEqual(symreg_row["final_symmetry_penalty"], 0.02)

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
                self.assertIn("mean_symmetry_twirled_test_accuracy", row)
                self.assertIn("mean_symmetry_twirled_raw_subset_accuracy", row)
                self.assertIn("mean_symmetry_twirled_num_correct_raw_subset", row)
                self.assertIn("symmetry_twirled_available", row)
                self.assertIn("mean_build_time_seconds", row)
                self.assertIn("mean_total_training_time_seconds", row)

            aggregated = aggregate_noisy_comparison_runs(results["runs"])
            self.assertEqual(len(aggregated), 2)
            self.assertEqual({row["noise_strength"] for row in aggregated}, {0.0, 0.01})

    def test_aggregation_keeps_noisy_qubit_and_overrotation_modes_separate(self) -> None:
        rows = [
            {
                "job_index": 0,
                "backend_name": "qiskit_mixed",
                "model_family": "su2_qcnn",
                "num_qubits": 5,
                "train_size": 4,
                "epochs": 10,
                "noise_model_name": "coherent_overrotation",
                "noise_strength": 0.01,
                "noise_application_scope": "selected_qubits",
                "noisy_qubit_index": 0,
                "coherent_overrotation_mode": "fixed",
                "coherent_overrotation_probability": 1.0,
                "coherent_overrotation_angle_std": 0.0,
                "train_accuracy": 0.8,
                "test_accuracy": 0.7,
                "train_loss": 0.5,
                "test_loss": 0.6,
                "classification_threshold": 0.5,
                "runtime_seconds": 1.0,
            },
            {
                "job_index": 1,
                "backend_name": "qiskit_mixed",
                "model_family": "su2_qcnn",
                "num_qubits": 5,
                "train_size": 4,
                "epochs": 10,
                "noise_model_name": "coherent_overrotation",
                "noise_strength": 0.01,
                "noise_application_scope": "selected_qubits",
                "noisy_qubit_index": 1,
                "coherent_overrotation_mode": "fixed",
                "coherent_overrotation_probability": 1.0,
                "coherent_overrotation_angle_std": 0.0,
                "train_accuracy": 0.81,
                "test_accuracy": 0.71,
                "train_loss": 0.51,
                "test_loss": 0.61,
                "classification_threshold": 0.5,
                "runtime_seconds": 1.0,
            },
            {
                "job_index": 2,
                "backend_name": "qiskit_mixed",
                "model_family": "su2_qcnn",
                "num_qubits": 5,
                "train_size": 4,
                "epochs": 10,
                "noise_model_name": "coherent_overrotation",
                "noise_strength": 0.01,
                "noise_application_scope": "selected_qubits",
                "noisy_qubit_index": 0,
                "coherent_overrotation_mode": "stochastic",
                "coherent_overrotation_probability": 0.5,
                "coherent_overrotation_angle_std": 0.0,
                "train_accuracy": 0.79,
                "test_accuracy": 0.69,
                "train_loss": 0.52,
                "test_loss": 0.62,
                "classification_threshold": 0.5,
                "runtime_seconds": 1.0,
            },
        ]

        aggregated = aggregate_noisy_comparison_runs(rows)

        self.assertEqual(len(aggregated), 3)
        keys = {
            (
                row["noisy_qubit_index"],
                row["coherent_overrotation_mode"],
                row["coherent_overrotation_probability"],
            )
            for row in aggregated
        }
        self.assertEqual(
            keys,
            {
                (0, "fixed", 1.0),
                (1, "fixed", 1.0),
                (0, "stochastic", 0.5),
            },
        )

    def test_aggregation_keeps_single_qubit_profiles_separate(self) -> None:
        rows = [
            {
                "job_index": 0,
                "backend_name": "qiskit_mixed",
                "model_family": "hea_qcnn",
                "num_qubits": 5,
                "train_size": 4,
                "epochs": 10,
                "noise_model_name": "amplitude_damping",
                "noise_strength": 0.02,
                "noise_primary_strength": 0.02,
                "noise_application_scope": "all",
                "noisy_qubit_index": None,
                "noisy_qubits": None,
                "single_qubit_error_profile": [0.02, 0.03],
                "single_qubit_depolarizing_error": 0.0,
                "two_qubit_depolarizing_error": 0.0,
                "amplitude_damping_gamma": 0.02,
                "phase_damping_gamma": 0.0,
                "coherent_overrotation_angle": 0.0,
                "coherent_overrotation_axis": "zz",
                "coherent_overrotation_mode": "fixed",
                "coherent_overrotation_probability": 1.0,
                "coherent_overrotation_angle_std": 0.0,
                "coherent_overrotation_seed": None,
                "pair_dependent_overrotation_angles": None,
                "train_accuracy": 0.8,
                "test_accuracy": 0.7,
                "train_loss": 0.5,
                "test_loss": 0.6,
                "classification_threshold": 0.5,
                "runtime_seconds": 1.0,
            },
            {
                "job_index": 1,
                "backend_name": "qiskit_mixed",
                "model_family": "hea_qcnn",
                "num_qubits": 5,
                "train_size": 4,
                "epochs": 10,
                "noise_model_name": "amplitude_damping",
                "noise_strength": 0.02,
                "noise_primary_strength": 0.02,
                "noise_application_scope": "all",
                "noisy_qubit_index": None,
                "noisy_qubits": None,
                "single_qubit_error_profile": [0.02, 0.04],
                "single_qubit_depolarizing_error": 0.0,
                "two_qubit_depolarizing_error": 0.0,
                "amplitude_damping_gamma": 0.02,
                "phase_damping_gamma": 0.0,
                "coherent_overrotation_angle": 0.0,
                "coherent_overrotation_axis": "zz",
                "coherent_overrotation_mode": "fixed",
                "coherent_overrotation_probability": 1.0,
                "coherent_overrotation_angle_std": 0.0,
                "coherent_overrotation_seed": None,
                "pair_dependent_overrotation_angles": None,
                "train_accuracy": 0.79,
                "test_accuracy": 0.69,
                "train_loss": 0.52,
                "test_loss": 0.62,
                "classification_threshold": 0.5,
                "runtime_seconds": 1.0,
            },
        ]

        aggregated = aggregate_noisy_comparison_runs(rows)

        self.assertEqual(len(aggregated), 2)

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
                        "eval_noise_strength": 0.0,
                        "seed": 0,
                        "noise_aware_training": True,
                        "training_noise_strengths": [0.0, 0.01],
                        "train_noise_strength_values": [0.0, 0.01],
                        "training_noise_sampling": "per_epoch",
                        "train_noise_sampling_mode": "per_epoch_random_choice",
                        "train_noise_includes_zero": True,
                        "training_noise_seed": 7,
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
                    "--compute-symmetry-twirled-evaluation",
                    "--num-symmetry-twirl-samples",
                    "2",
                    "--symmetry-twirl-seed",
                    "123",
                    "--num-state-samples-for-twirled-evaluation",
                    "1",
                    "--noise-aware-training",
                    "--train-noise-strength-values",
                    "0.0",
                    "0.01",
                    "--train-noise-sampling-mode",
                    "per_epoch_random_choice",
                    "--training-noise-seed",
                    "7",
                    "--symmetry-regularization",
                    "--symmetry-regularization-weight",
                    "0.1",
                    "--num-symmetry-regularization-samples",
                    "1",
                    "--symmetry-regularization-frequency",
                    "1",
                    "--symmetry-regularization-state-samples",
                    "1",
                    "--symmetry-regularization-seed",
                    "9",
                    "--aggregate-only",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "summary.csv").exists())
            rows = list(csv.DictReader((output_dir / "summary.csv").read_text().splitlines()))
            self.assertEqual(rows[0]["train_noise_strength_values"], "[0.0, 0.01]")
            self.assertEqual(rows[0]["train_noise_sampling_mode"], "per_epoch_random_choice")
            self.assertEqual(rows[0]["train_noise_includes_zero"], "True")
            self.assertEqual(rows[0]["eval_noise_strength"], "0.0")
            config_json = json.loads((output_dir / "noisy_comparison_config.json").read_text())
            self.assertEqual(config_json["symmetry_regularization_weight"], 0.1)
            self.assertFalse(config_json["symmetry_regularization_beta_sweep"])
            self.assertEqual(config_json["resolved_symmetry_regularization_beta_values"], [0.1])

    def test_cli_accepts_symmetry_regularization_beta_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "cli_beta_values"
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
                json.dumps(self._minimal_run_row(output_dir=str(run_dir.resolve())), indent=2, sort_keys=True)
                + "\n"
            )

            exit_code = cli_main(
                [
                    "run-noisy-comparison",
                    "--symmetry-regularization",
                    "--symmetry-regularization-beta-values",
                    "0.0",
                    "0.01",
                    "0.1",
                    "--aggregate-only",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            config_json = json.loads((output_dir / "noisy_comparison_config.json").read_text())
            self.assertTrue(config_json["symmetry_regularization"])
            self.assertEqual(config_json["symmetry_regularization_beta_values"], [0.0, 0.01, 0.1])
            self.assertEqual(config_json["resolved_symmetry_regularization_beta_values"], [0.0, 0.01, 0.1])

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
            compute_symmetry_twirled_evaluation=True,
            num_symmetry_twirl_samples=1,
            symmetry_twirl_seed=0,
            num_state_samples_for_twirled_evaluation=1,
            noise_aware_training=True,
            training_noise_strengths=(0.0, 0.01),
            training_noise_seed=0,
            symmetry_regularization=True,
            symmetry_regularization_weight=0.01,
            num_symmetry_regularization_samples=1,
            symmetry_regularization_state_samples=1,
            symmetry_regularization_seed=0,
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
            self.assertIn("symmetry_twirled_available", run)
            self.assertIn("symmetry_twirled_test_accuracy", run)
            self.assertIn("symmetry_twirled_raw_subset_accuracy", run)
            self.assertIn("symmetry_twirled_num_correct_raw_subset", run)
            self.assertIn("symmetry_twirled_num_correct_twirled_subset", run)
            self.assertEqual(run["num_state_samples_for_twirled_evaluation"], 1)
            if run["symmetry_twirled_available"]:
                self.assertEqual(run["symmetry_twirled_subset_size"], 1)
            self.assertTrue(run["noise_aware_training"])
            self.assertEqual(run["training_noise_strengths"], [0.0, 0.01])
            self.assertEqual(len(run["training_noise_schedule"]), 1)
            self.assertTrue(run["symmetry_regularization"])
            self.assertTrue(run["symmetry_regularization_enabled"])
            self.assertEqual(run["symmetry_regularization_beta"], 0.01)
            self.assertEqual(run["symmetry_regularization_weight"], 0.01)
            self.assertIn("symmetry_regularization_note", run)
            self.assertIn("final_symmetry_penalty", run)
            self.assertTrue((run_output_dir / "metrics.json").exists())
            self.assertTrue((run_output_dir / "best_parameters.npy").exists())
            self.assertTrue((run_output_dir / "noisy_job_config.json").exists())
            self.assertTrue((run_output_dir / "runtime_profile.json").exists())
            self.assertTrue((run_output_dir / "runtime_breakdown.json").exists())
            self.assertTrue((run_output_dir / "noisy_run_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
