from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eqnn.experiments import (
    PaperDatasetConfig,
    PaperReproductionConfig,
    generate_paper_dataset,
    paper_test_ratios,
    paper_training_ratios,
    run_paper_reproduction_suite,
)


class PaperReproductionTests(unittest.TestCase):
    def test_paper_training_ratios_span_both_sides_of_the_transition(self) -> None:
        ratios = paper_training_ratios(PaperDatasetConfig(num_qubits=4, train_size=2))
        np.testing.assert_allclose(ratios, np.asarray([0.5, 1.5], dtype=np.float64))

    def test_paper_dataset_uses_exact_train_size_and_dense_phase_grid(self) -> None:
        dataset = generate_paper_dataset(
            PaperDatasetConfig(
                num_qubits=4,
                train_size=4,
                dense_test_points=11,
            )
        )

        self.assertEqual(len(dataset.train), 4)
        self.assertEqual(len(dataset.test), 10)
        self.assertTrue(np.all(dataset.train.coupling_ratios < 2.0))
        self.assertTrue(np.all(dataset.train.coupling_ratios != 1.0))
        self.assertEqual(dataset.metadata["protocol"], "paper_reproduction_v1")

    def test_paper_reproduction_suite_writes_summary_and_phase_diagram_outputs(self) -> None:
        config = PaperReproductionConfig(
            num_qubits=4,
            train_sizes=(2,),
            random_seeds=(0,),
            epochs=1,
            dense_test_points=11,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "paper"
            results = run_paper_reproduction_suite(config, output_dir)

            self.assertEqual(len(results["summary"]), 1)
            self.assertEqual(len(results["runs"]), 1)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "runs.json").exists())

            summary_rows = list(csv.DictReader((output_dir / "summary.csv").read_text().splitlines()))
            self.assertEqual(len(summary_rows), 1)
            phase_diagram_path = Path(summary_rows[0]["phase_diagram_path"])
            self.assertTrue(phase_diagram_path.exists())

            with np.load(phase_diagram_path, allow_pickle=False) as data:
                self.assertIn("mean_probabilities", data.files)
                self.assertIn("variance_probabilities", data.files)
                self.assertIn("coupling_ratios", data.files)
