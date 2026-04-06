"""Simple noise-configuration objects for mixed-state simulation backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseConfig:
    """Simple explicit noise configuration for mixed-state backend experiments."""

    noise_model_name: str = "none"
    single_qubit_depolarizing_error: float = 0.0
    two_qubit_depolarizing_error: float = 0.0
    amplitude_damping_gamma: float = 0.0
    phase_damping_gamma: float = 0.0
    readout_error_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.noise_model_name not in {"none", "depolarizing", "amplitude_damping", "phase_damping"}:
            raise ValueError(
                "noise_model_name must be 'none', 'depolarizing', 'amplitude_damping', or 'phase_damping'"
            )
        for field_name in (
            "single_qubit_depolarizing_error",
            "two_qubit_depolarizing_error",
            "amplitude_damping_gamma",
            "phase_damping_gamma",
            "readout_error_probability",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")

    @property
    def is_noiseless(self) -> bool:
        return (
            self.noise_model_name == "none"
            and self.single_qubit_depolarizing_error == 0.0
            and self.two_qubit_depolarizing_error == 0.0
            and self.amplitude_damping_gamma == 0.0
            and self.phase_damping_gamma == 0.0
            and self.readout_error_probability == 0.0
        )

    def apply_readout_error(self, probability: float) -> float:
        error_probability = float(self.readout_error_probability)
        if error_probability <= 0.0:
            return float(probability)
        return float((1.0 - error_probability) * probability + error_probability * (1.0 - probability))
