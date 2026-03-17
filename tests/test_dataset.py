from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eqnn.datasets.heisenberg import (
    HeisenbergDatasetConfig,
    generate_dataset,
    phase_label_from_ratio,
)
from eqnn.datasets.io import load_dataset_bundle, save_dataset_bundle
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
