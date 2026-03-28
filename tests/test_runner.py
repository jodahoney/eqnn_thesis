from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eqnn.datasets.cache import load_or_generate_cached_dataset
from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.experiments.runner import (
    ExperimentConfig,
    load_or_generate_dataset,
    run_training_experiment,
)
from eqnn.training import TrainingConfig
from eqnn.utils.timing import RuntimeProfile


class RunnerTimingAndCacheTests(unittest.TestCase):
    def _small_dataset_config(self, *, split_seed: int = 0) -> HeisenbergDatasetConfig:
        return HeisenbergDatasetConfig(
            num_qubits=4,
            ratio_min=0.8,
            ratio_max=1.2,
            num_points=7,
            train_fraction=0.8,
            critical_ratio=1.0,
            exclusion_window=0.02,
            labeling_strategy="ratio_threshold",
            split_seed=split_seed,
        )

    def _small_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            epochs=2,
            learning_rate=0.05,
            batch_size=2,
            gradient_backend="exact",
            random_seed=0,
        )

    def test_run_training_experiment_records_runtime_profile_and_writes_artifacts(self) -> None:
        profile = RuntimeProfile()

        dataset = generate_dataset(
            self._small_dataset_config(),
            profile=profile,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "runner_timing_smoke"

            result = run_training_experiment(
                dataset=dataset,
                experiment_config=ExperimentConfig(
                    model_family="baseline_qcnn",
                    num_qubits=4,
                    pooling_mode="partial_trace",
                    readout_mode="swap",
                ),
                training_config=self._small_training_config(),
                output_dir=output_dir,
                experiment_name="runner_timing_smoke",
                profile=profile,
            )

            self.assertIn("runtime_profile", result)
            runtime_profile = result["runtime_profile"]
            self.assertIsInstance(runtime_profile, dict)
            self.assertTrue(runtime_profile)

            expected_keys = {
                "dataset.total",
                "experiment.build_model",
                "experiment.build_trainer",
                "experiment.train_fit",
                "experiment.final_train_evaluate",
                "experiment.final_test_evaluate",
                "experiment.train_predict",
                "experiment.test_predict",
                "experiment.write_artifacts",
            }
            self.assertTrue(expected_keys.issubset(runtime_profile.keys()))

            for key in expected_keys:
                self.assertGreaterEqual(runtime_profile[key]["total_seconds"], 0.0)
                self.assertGreaterEqual(runtime_profile[key]["count"], 1)

            self.assertIn("train_metrics", result)
            self.assertIn("test_metrics", result)
            self.assertIn("output_dir", result)

            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "best_parameters.npy").exists())
            self.assertTrue((output_dir / "train_predictions.npz").exists())
            self.assertTrue((output_dir / "test_predictions.npz").exists())

            saved_metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertIn("runtime_profile", saved_metrics)
            self.assertTrue(expected_keys.issubset(saved_metrics["runtime_profile"].keys()))

    def test_load_or_generate_dataset_profiles_generation_path(self) -> None:
        profile = RuntimeProfile()

        bundle = load_or_generate_dataset(
            dataset_config=self._small_dataset_config(split_seed=1),
            profile=profile,
        )

        self.assertGreater(len(bundle.train), 0)
        self.assertGreater(len(bundle.test), 0)

        summary = profile.summary()
        self.assertIn("dataset.generate_bundle", summary)
        self.assertIn("dataset.total", summary)
        self.assertGreaterEqual(summary["dataset.generate_bundle"]["count"], 1)
        self.assertGreaterEqual(summary["dataset.total"]["count"], 1)

    def test_cached_dataset_miss_then_hit(self) -> None:
        config = self._small_dataset_config(split_seed=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            miss_profile = RuntimeProfile()
            bundle_1, cache_path_1, cache_hit_1 = load_or_generate_cached_dataset(
                config,
                cache_dir=cache_dir,
                profile=miss_profile,
            )

            self.assertFalse(cache_hit_1)
            self.assertTrue(cache_path_1.exists())
            self.assertTrue((cache_path_1 / "train.npz").exists())
            self.assertTrue((cache_path_1 / "test.npz").exists())
            self.assertTrue((cache_path_1 / "metadata.json").exists())

            miss_summary = miss_profile.summary()
            self.assertIn("cache.dataset_generate", miss_summary)
            self.assertIn("cache.dataset_save", miss_summary)
            self.assertIn("dataset.total", miss_summary)
            self.assertNotIn("cache.dataset_load", miss_summary)

            hit_profile = RuntimeProfile()
            bundle_2, cache_path_2, cache_hit_2 = load_or_generate_cached_dataset(
                config,
                cache_dir=cache_dir,
                profile=hit_profile,
            )

            self.assertTrue(cache_hit_2)
            self.assertEqual(cache_path_1, cache_path_2)

            hit_summary = hit_profile.summary()
            self.assertIn("cache.dataset_load", hit_summary)
            self.assertNotIn("cache.dataset_generate", hit_summary)

            self.assertEqual(bundle_1.train.states.shape, bundle_2.train.states.shape)
            self.assertEqual(bundle_1.test.states.shape, bundle_2.test.states.shape)
            self.assertEqual(bundle_1.metadata["num_total_samples"], bundle_2.metadata["num_total_samples"])

    def test_cached_dataset_force_rebuild(self) -> None:
        config = self._small_dataset_config(split_seed=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            _, cache_path_1, cache_hit_1 = load_or_generate_cached_dataset(
                config,
                cache_dir=cache_dir,
                profile=RuntimeProfile(),
            )
            self.assertFalse(cache_hit_1)
            self.assertTrue(cache_path_1.exists())

            rebuild_profile = RuntimeProfile()
            _, cache_path_2, cache_hit_2 = load_or_generate_cached_dataset(
                config,
                cache_dir=cache_dir,
                profile=rebuild_profile,
                force_rebuild=True,
            )

            self.assertFalse(cache_hit_2)
            self.assertEqual(cache_path_1, cache_path_2)

            rebuild_summary = rebuild_profile.summary()
            self.assertIn("cache.dataset_generate", rebuild_summary)
            self.assertIn("cache.dataset_save", rebuild_summary)
            self.assertNotIn("cache.dataset_load", rebuild_summary)

    def test_runner_loads_dataset_from_saved_directory_with_profile(self) -> None:
        config = self._small_dataset_config(split_seed=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"

            generated = generate_dataset(config, profile=RuntimeProfile())

            from eqnn.datasets.io import save_dataset_bundle
            save_dataset_bundle(generated, dataset_dir, profile=RuntimeProfile())

            load_profile = RuntimeProfile()
            loaded = load_or_generate_dataset(
                dataset_dir=dataset_dir,
                profile=load_profile,
            )

            self.assertEqual(generated.train.states.shape, loaded.train.states.shape)
            self.assertEqual(generated.test.states.shape, loaded.test.states.shape)

            summary = load_profile.summary()
            self.assertIn("dataset.load_bundle", summary)
            self.assertIn("io.load_dataset_bundle", summary)
            self.assertNotIn("dataset.generate_bundle", summary)


if __name__ == "__main__":
    unittest.main()