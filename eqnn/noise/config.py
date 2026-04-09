"""Explicit noise-configuration objects for mixed-state simulation backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


SUPPORTED_NOISE_MODELS = (
    "none",
    "depolarizing",
    "phase_damping",
    "amplitude_damping",
    "coherent_overrotation",
)

_NOISE_MODEL_ALIASES = {
    "dephasing": "phase_damping",
    "overrotation": "coherent_overrotation",
}


@dataclass(frozen=True)
class NoiseConfig:
    """Explicit noise configuration for mixed-state backend experiments."""

    noise_model_name: str = "none"
    single_qubit_depolarizing_error: float = 0.0
    two_qubit_depolarizing_error: float = 0.0
    amplitude_damping_gamma: float = 0.0
    phase_damping_gamma: float = 0.0
    coherent_overrotation_angle: float = 0.0
    coherent_overrotation_axis: str = "zz"
    readout_error_probability: float = 0.0

    def __post_init__(self) -> None:
        canonical_name = _NOISE_MODEL_ALIASES.get(self.noise_model_name, self.noise_model_name)
        object.__setattr__(self, "noise_model_name", canonical_name)

        if canonical_name not in SUPPORTED_NOISE_MODELS:
            raise ValueError(
                "noise_model_name must be one of "
                f"{SUPPORTED_NOISE_MODELS} (aliases: {tuple(_NOISE_MODEL_ALIASES)})"
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
        if not isfinite(float(self.coherent_overrotation_angle)):
            raise ValueError("coherent_overrotation_angle must be finite")
        if self.coherent_overrotation_axis not in {"zz"}:
            raise ValueError("coherent_overrotation_axis must currently be 'zz'")

    @property
    def is_noiseless(self) -> bool:
        return (
            self.noise_model_name == "none"
            and self.single_qubit_depolarizing_error == 0.0
            and self.two_qubit_depolarizing_error == 0.0
            and self.amplitude_damping_gamma == 0.0
            and self.phase_damping_gamma == 0.0
            and self.coherent_overrotation_angle == 0.0
            and self.readout_error_probability == 0.0
        )

    @property
    def primary_strength(self) -> float:
        if self.noise_model_name == "depolarizing":
            return float(
                max(self.single_qubit_depolarizing_error, self.two_qubit_depolarizing_error)
            )
        if self.noise_model_name == "amplitude_damping":
            return float(self.amplitude_damping_gamma)
        if self.noise_model_name == "phase_damping":
            return float(self.phase_damping_gamma)
        if self.noise_model_name == "coherent_overrotation":
            return float(abs(self.coherent_overrotation_angle))
        return 0.0

    def parameter_metadata(self) -> dict[str, float | str]:
        return {
            "single_qubit_depolarizing_error": float(self.single_qubit_depolarizing_error),
            "two_qubit_depolarizing_error": float(self.two_qubit_depolarizing_error),
            "amplitude_damping_gamma": float(self.amplitude_damping_gamma),
            "phase_damping_gamma": float(self.phase_damping_gamma),
            "coherent_overrotation_angle": float(self.coherent_overrotation_angle),
            "coherent_overrotation_axis": str(self.coherent_overrotation_axis),
            "readout_error_probability": float(self.readout_error_probability),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "noise_model_name": str(self.noise_model_name),
            "primary_strength": float(self.primary_strength),
            "is_noiseless": bool(self.is_noiseless),
            "parameters": self.parameter_metadata(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_readout_error(self, probability: float) -> float:
        error_probability = float(self.readout_error_probability)
        if error_probability <= 0.0:
            return float(probability)
        return float((1.0 - error_probability) * probability + error_probability * (1.0 - probability))


def noise_config_from_strength(
    noise_model_name: str,
    noise_strength: float,
    *,
    readout_error_probability: float = 0.0,
) -> NoiseConfig:
    canonical_name = _NOISE_MODEL_ALIASES.get(noise_model_name, noise_model_name)
    strength = float(noise_strength)

    if canonical_name in {"none", "depolarizing", "amplitude_damping", "phase_damping"} and not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"noise_strength must lie in [0, 1] for noise_model_name='{canonical_name}'"
        )
    if canonical_name == "coherent_overrotation" and not isfinite(strength):
        raise ValueError("noise_strength must be finite for noise_model_name='coherent_overrotation'")
    if canonical_name == "none" and strength != 0.0:
        raise ValueError("noise_strength must be 0.0 when noise_model_name='none'")

    if canonical_name == "none":
        return NoiseConfig(
            noise_model_name="none",
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "depolarizing":
        return NoiseConfig(
            noise_model_name="depolarizing",
            single_qubit_depolarizing_error=strength,
            two_qubit_depolarizing_error=strength,
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "amplitude_damping":
        return NoiseConfig(
            noise_model_name="amplitude_damping",
            amplitude_damping_gamma=strength,
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "phase_damping":
        return NoiseConfig(
            noise_model_name="phase_damping",
            phase_damping_gamma=strength,
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "coherent_overrotation":
        return NoiseConfig(
            noise_model_name="coherent_overrotation",
            coherent_overrotation_angle=strength,
            readout_error_probability=readout_error_probability,
        )
    raise ValueError(
        "noise_model_name must be one of "
        f"{SUPPORTED_NOISE_MODELS} (aliases: {tuple(_NOISE_MODEL_ALIASES)})"
    )
