from __future__ import annotations

import unittest

from eqnn.noise import SUPPORTED_NOISE_MODELS, NoiseConfig, noise_config_from_strength


class NoiseConfigTests(unittest.TestCase):
    def test_noise_config_from_strength_canonicalizes_dephasing_alias(self) -> None:
        config = noise_config_from_strength("dephasing", 0.05)

        self.assertEqual(config.noise_model_name, "phase_damping")
        self.assertAlmostEqual(config.phase_damping_gamma, 0.05)
        self.assertEqual(config.to_metadata()["noise_model_name"], "phase_damping")
        self.assertAlmostEqual(float(config.to_metadata()["primary_strength"]), 0.05)

    def test_coherent_overrotation_is_supported_and_serializable(self) -> None:
        config = noise_config_from_strength("coherent_overrotation", 0.1)

        self.assertEqual(config.noise_model_name, "coherent_overrotation")
        self.assertAlmostEqual(config.coherent_overrotation_angle, 0.1)
        self.assertEqual(config.coherent_overrotation_axis, "zz")
        self.assertIn("coherent_overrotation", SUPPORTED_NOISE_MODELS)
        self.assertEqual(config.to_dict()["coherent_overrotation_axis"], "zz")

    def test_probability_like_noise_strength_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must lie in \\[0, 1\\]"):
            noise_config_from_strength("depolarizing", 1.2)

    def test_noise_config_rejects_invalid_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "coherent_overrotation_axis"):
            NoiseConfig(
                noise_model_name="coherent_overrotation",
                coherent_overrotation_angle=0.1,
                coherent_overrotation_axis="x",
            )


if __name__ == "__main__":
    unittest.main()
