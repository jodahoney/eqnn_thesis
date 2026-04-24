from __future__ import annotations

import unittest

from eqnn.backends.qiskit_mixed import QiskitMixedStateBackend
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

    def test_noise_config_accepts_extended_coherent_modes(self) -> None:
        for mode in ("fixed", "stochastic", "random_angle"):
            config = NoiseConfig(
                noise_model_name="coherent_overrotation",
                coherent_overrotation_angle=0.1,
                coherent_overrotation_mode=mode,
            )
            self.assertEqual(config.coherent_overrotation_mode, mode)

    def test_noise_config_rejects_invalid_coherent_probability_and_std(self) -> None:
        with self.assertRaisesRegex(ValueError, "coherent_overrotation_probability"):
            NoiseConfig(
                noise_model_name="coherent_overrotation",
                coherent_overrotation_angle=0.1,
                coherent_overrotation_mode="stochastic",
                coherent_overrotation_probability=1.5,
            )

        with self.assertRaisesRegex(ValueError, "coherent_overrotation_angle_std"):
            NoiseConfig(
                noise_model_name="coherent_overrotation",
                coherent_overrotation_angle=0.1,
                coherent_overrotation_mode="random_angle",
                coherent_overrotation_angle_std=-0.1,
            )

    def test_noise_config_selected_qubits_validation(self) -> None:
        config = NoiseConfig(
            noise_model_name="amplitude_damping",
            amplitude_damping_gamma=0.1,
            noise_application_scope="selected_qubits",
            noisy_qubits=(0, 2),
        )
        self.assertEqual(config.noisy_qubits, (0, 2))

        with self.assertRaisesRegex(ValueError, "noisy_qubits must be provided"):
            NoiseConfig(
                noise_model_name="phase_damping",
                phase_damping_gamma=0.1,
                noise_application_scope="selected_qubits",
            )

    def test_selected_qubits_outside_effective_register_are_skipped(self) -> None:
        backend = object.__new__(QiskitMixedStateBackend)
        backend.noise_config = NoiseConfig(
            noise_model_name="amplitude_damping",
            amplitude_damping_gamma=0.1,
            noise_application_scope="selected_qubits",
            noisy_qubits=(4,),
        )

        self.assertEqual(backend._target_noise_sites(active_qubits=(0, 1), num_qubits=5), (4,))
        self.assertEqual(backend._target_noise_sites(active_qubits=(0, 1), num_qubits=2), ())

    def test_noise_metadata_documents_profile_override_and_pair_indexing(self) -> None:
        config = NoiseConfig(
            noise_model_name="coherent_overrotation",
            coherent_overrotation_angle=0.1,
            coherent_overrotation_mode="pair_dependent",
            pair_dependent_overrotation_angles=(0.1, 0.2),
            single_qubit_error_profile=(0.03, 0.04),
        )

        metadata = config.parameter_metadata()

        self.assertEqual(metadata["single_qubit_error_profile_mode"], "override_scalar_strength")
        self.assertEqual(metadata["pair_dependent_overrotation_indexing"], "active_pair_order")

    def test_probability_like_noise_strength_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must lie in \\[0, 1\\]"):
            noise_config_from_strength("depolarizing", 1.2)

    def test_noise_config_from_strength_covers_supported_models(self) -> None:
        depolarizing = noise_config_from_strength("depolarizing", 0.03)
        amplitude = noise_config_from_strength("amplitude_damping", 0.04)
        phase = noise_config_from_strength("phase_damping", 0.05)
        coherent = noise_config_from_strength(
            "coherent_overrotation",
            0.1,
            coherent_overrotation_mode="stochastic",
            coherent_overrotation_probability=0.25,
        )

        self.assertAlmostEqual(depolarizing.single_qubit_depolarizing_error, 0.03)
        self.assertAlmostEqual(amplitude.amplitude_damping_gamma, 0.04)
        self.assertAlmostEqual(phase.phase_damping_gamma, 0.05)
        self.assertEqual(coherent.coherent_overrotation_mode, "stochastic")
        self.assertAlmostEqual(coherent.coherent_overrotation_probability, 0.25)

    def test_noise_config_rejects_invalid_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "coherent_overrotation_axis"):
            NoiseConfig(
                noise_model_name="coherent_overrotation",
                coherent_overrotation_angle=0.1,
                coherent_overrotation_axis="x",
            )


if __name__ == "__main__":
    unittest.main()
