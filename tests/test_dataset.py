from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eqnn.datasets.heisenberg import (
    HeisenbergDatasetConfig,
    generate_dataset,
    phase_label_from_partial_reflection,
    phase_label_from_ratio,
)
from eqnn.datasets.io import load_dataset_bundle, save_dataset_bundle
from eqnn.physics.observables import dimerization_feature
from eqnn.physics.quantum import as_density_matrix
from eqnn.physics.spin import sparse

SCIPY_AVAILABLE = sparse is not None


class PhaseLabelTests(unittest.TestCase):
    def test_phase_label_assignment(self) -> None:
        self.assertEqual(
            phase_label_from_ratio(0.8, critical_ratio=1.0, exclusion_window=0.05),
            0,
        )
        self.assertEqual(
            phase_label_from_ratio(1.2, critical_ratio=1.0, exclusion_window=0.05),
            1,
        )

    def test_phase_label_excludes_transition_window(self) -> None:
        with self.assertRaises(ValueError):
            phase_label_from_ratio(1.02, critical_ratio=1.0, exclusion_window=0.05)

    def test_partial_reflection_label_assignment(self) -> None:
        self.assertEqual(
            phase_label_from_partial_reflection(-0.12, diagnostic_window=0.05),
            0,
        )
        self.assertEqual(
            phase_label_from_partial_reflection(0.12, diagnostic_window=0.05),
            1,
        )

    def test_partial_reflection_label_excludes_small_scores(self) -> None:
        with self.assertRaises(ValueError):
            phase_label_from_partial_reflection(0.01, diagnostic_window=0.05)


class DatasetGenerationTests(unittest.TestCase):
    def test_dataset_generation_returns_both_splits(self) -> None:
        config = HeisenbergDatasetConfig(
            num_qubits=4,
            ratio_min=0.4,
            ratio_max=1.6,
            num_points=13,
            train_fraction=0.75,
            exclusion_window=0.05,
            split_seed=123,
        )
        bundle = generate_dataset(config)

        self.assertGreater(len(bundle.train), 0)
        self.assertGreater(len(bundle.test), 0)
        self.assertEqual(bundle.train.states.shape[1], 16)
        self.assertEqual(bundle.test.states.shape[1], 16)

        labels = np.concatenate([bundle.train.labels, bundle.test.labels])
        self.assertSetEqual(set(labels.tolist()), {0, 1})

    def test_dataset_round_trip(self) -> None:
        config = HeisenbergDatasetConfig(
            num_qubits=4,
            ratio_min=0.4,
            ratio_max=1.6,
            num_points=13,
            split_seed=7,
        )
        bundle = generate_dataset(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_dataset_bundle(bundle, output_dir)
            loaded = load_dataset_bundle(output_dir)

            np.testing.assert_allclose(bundle.train.states, loaded.train.states)
            np.testing.assert_array_equal(bundle.train.labels, loaded.train.labels)
            np.testing.assert_allclose(
                bundle.test.coupling_ratios,
                loaded.test.coupling_ratios,
            )
            self.assertIsNotNone(loaded.train.diagnostics)
            np.testing.assert_allclose(
                bundle.train.diagnostics["dimerization_feature"],
                loaded.train.diagnostics["dimerization_feature"],
            )
            self.assertEqual(bundle.metadata["num_qubits"], loaded.metadata["num_qubits"])

    def test_dataset_records_solver_metadata(self) -> None:
        bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=4,
                ratio_min=0.4,
                ratio_max=1.6,
                num_points=13,
                eigensolver="dense",
            )
        )
        self.assertEqual(bundle.metadata["eigensolver_requested"], "dense")
        self.assertEqual(bundle.metadata["eigensolver_resolved"], "dense")

    def test_ratio_threshold_labels_align_with_a_dimerization_sanity_check(self) -> None:
        bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=6,
                ratio_min=0.4,
                ratio_max=1.6,
                num_points=13,
                split_seed=11,
            )
        )

        states = np.concatenate([bundle.train.states, bundle.test.states], axis=0)
        labels = np.concatenate([bundle.train.labels, bundle.test.labels], axis=0)
        ratios = np.concatenate([bundle.train.coupling_ratios, bundle.test.coupling_ratios], axis=0)
        features = np.asarray(
            [
                dimerization_feature(as_density_matrix(state), num_qubits=6)
                for state in states
            ],
            dtype=np.float64,
        )

        self.assertGreater(float(np.mean(features[labels == 1])), float(np.mean(features[labels == 0])) + 0.2)
        self.assertGreater(float(np.corrcoef(ratios, features)[0, 1]), 0.99)

    def test_dataset_records_phase_diagnostics(self) -> None:
        bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=6,
                ratio_min=0.4,
                ratio_max=1.6,
                num_points=13,
                split_seed=5,
            )
        )

        self.assertIn("diagnostics", bundle.metadata)
        self.assertEqual(bundle.metadata["partial_reflection_pairs"], 2)
        self.assertIsNotNone(bundle.train.diagnostics)
        self.assertIn("partial_reflection_real", bundle.train.diagnostics)
        self.assertIn("dimerization_feature", bundle.train.diagnostics)
        self.assertIn("reference_phase_labels", bundle.metadata)

    def test_partial_reflection_labeling_tracks_the_finite_size_phase_shift(self) -> None:
        reflection_bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=6,
                ratio_min=0.4,
                ratio_max=1.6,
                num_points=13,
                split_seed=17,
                labeling_strategy="partial_reflection",
                diagnostic_window=0.02,
            )
        )

        ratios = np.concatenate(
            [reflection_bundle.train.coupling_ratios, reflection_bundle.test.coupling_ratios]
        )
        labels = np.concatenate([reflection_bundle.train.labels, reflection_bundle.test.labels])
        calibrated_scores = np.concatenate(
            [
                reflection_bundle.train.diagnostics["partial_reflection_calibrated"],
                reflection_bundle.test.diagnostics["partial_reflection_calibrated"],
            ]
        )
        order = np.argsort(ratios)

        sorted_ratios = ratios[order]
        sorted_labels = labels[order]
        sorted_scores = calibrated_scores[order]

        np.testing.assert_array_equal(sorted_labels[:8], np.zeros(8, dtype=np.int64))
        np.testing.assert_array_equal(sorted_labels[8:], np.ones(4, dtype=np.int64))
        self.assertLess(sorted_scores[7], 0.0)
        self.assertGreater(sorted_scores[8], 0.0)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for sparse tests")
    def test_sparse_dataset_generation_works(self) -> None:
        bundle = generate_dataset(
            HeisenbergDatasetConfig(
                num_qubits=4,
                ratio_min=0.4,
                ratio_max=1.6,
                num_points=9,
                eigensolver="sparse",
            )
        )
        self.assertGreater(len(bundle.train), 0)
        self.assertEqual(bundle.metadata["eigensolver_resolved"], "sparse")
