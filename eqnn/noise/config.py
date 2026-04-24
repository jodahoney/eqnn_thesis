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
    coherent_overrotation_mode: str = "fixed"
    coherent_overrotation_probability: float = 1.0
    coherent_overrotation_angle_std: float = 0.0
    coherent_overrotation_seed: int | None = None
    pair_dependent_overrotation_angles: tuple[float, ...] | None = None
    noise_application_scope: str = "active"
    noisy_qubits: tuple[int, ...] | None = None
    single_qubit_error_profile: tuple[float, ...] | None = None
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
            "coherent_overrotation_probability",
            "readout_error_probability",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
        if not isfinite(float(self.coherent_overrotation_angle)):
            raise ValueError("coherent_overrotation_angle must be finite")
        if not isfinite(float(self.coherent_overrotation_angle_std)) or float(self.coherent_overrotation_angle_std) < 0.0:
            raise ValueError("coherent_overrotation_angle_std must be finite and non-negative")
        if self.coherent_overrotation_axis not in {"zz"}:
            raise ValueError("coherent_overrotation_axis must currently be 'zz'")
        if self.coherent_overrotation_mode not in {
            "fixed",
            "stochastic",
            "random_angle",
            "pair_dependent",
        }:
            raise ValueError(
                "coherent_overrotation_mode must be 'fixed', 'stochastic', 'random_angle', or 'pair_dependent'"
            )
        if self.noise_application_scope not in {"active", "all", "selected_qubits"}:
            raise ValueError("noise_application_scope must be 'active', 'all', or 'selected_qubits'")
        if self.noisy_qubits is not None:
            normalized_qubits = tuple(int(site) for site in self.noisy_qubits)
            if any(site < 0 for site in normalized_qubits):
                raise ValueError("noisy_qubits must contain non-negative integers")
            object.__setattr__(self, "noisy_qubits", normalized_qubits)
        if self.noise_application_scope == "selected_qubits" and not self.noisy_qubits:
            raise ValueError("noisy_qubits must be provided when noise_application_scope='selected_qubits'")
        if self.single_qubit_error_profile is not None:
            normalized_profile = tuple(float(value) for value in self.single_qubit_error_profile)
            for value in normalized_profile:
                if not 0.0 <= value <= 1.0:
                    raise ValueError("single_qubit_error_profile values must lie in [0, 1]")
            object.__setattr__(self, "single_qubit_error_profile", normalized_profile)
        if self.pair_dependent_overrotation_angles is not None:
            normalized_angles = tuple(float(value) for value in self.pair_dependent_overrotation_angles)
            if any(not isfinite(value) for value in normalized_angles):
                raise ValueError("pair_dependent_overrotation_angles must be finite")
            object.__setattr__(self, "pair_dependent_overrotation_angles", normalized_angles)
        if self.coherent_overrotation_seed is not None:
            object.__setattr__(self, "coherent_overrotation_seed", int(self.coherent_overrotation_seed))

    @property
    def is_noiseless(self) -> bool:
        coherent_is_noiseless = self._coherent_noise_is_noiseless()
        single_qubit_profile_is_noiseless = (
            self.single_qubit_error_profile is None
            or all(float(value) == 0.0 for value in self.single_qubit_error_profile)
        )
        return (
            self.single_qubit_depolarizing_error == 0.0
            and self.two_qubit_depolarizing_error == 0.0
            and self.amplitude_damping_gamma == 0.0
            and self.phase_damping_gamma == 0.0
            and single_qubit_profile_is_noiseless
            and coherent_is_noiseless
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

    def parameter_metadata(self) -> dict[str, Any]:
        return {
            "single_qubit_depolarizing_error": float(self.single_qubit_depolarizing_error),
            "two_qubit_depolarizing_error": float(self.two_qubit_depolarizing_error),
            "amplitude_damping_gamma": float(self.amplitude_damping_gamma),
            "phase_damping_gamma": float(self.phase_damping_gamma),
            "coherent_overrotation_angle": float(self.coherent_overrotation_angle),
            "coherent_overrotation_axis": str(self.coherent_overrotation_axis),
            "coherent_overrotation_mode": str(self.coherent_overrotation_mode),
            "coherent_overrotation_probability": float(self.coherent_overrotation_probability),
            "coherent_overrotation_angle_std": float(self.coherent_overrotation_angle_std),
            "coherent_overrotation_seed": self.coherent_overrotation_seed,
            "pair_dependent_overrotation_angles": (
                None
                if self.pair_dependent_overrotation_angles is None
                else tuple(float(value) for value in self.pair_dependent_overrotation_angles)
            ),
            "pair_dependent_overrotation_indexing": (
                "active_pair_order" if self.pair_dependent_overrotation_angles is not None else None
            ),
            "noise_application_scope": str(self.noise_application_scope),
            "noisy_qubits": None if self.noisy_qubits is None else tuple(int(site) for site in self.noisy_qubits),
            "single_qubit_error_profile": (
                None
                if self.single_qubit_error_profile is None
                else tuple(float(value) for value in self.single_qubit_error_profile)
            ),
            "single_qubit_error_profile_mode": (
                "override_scalar_strength" if self.single_qubit_error_profile is not None else None
            ),
            "readout_error_probability": float(self.readout_error_probability),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "noise_model_name": str(self.noise_model_name),
            "primary_strength": float(self.primary_strength),
            "is_noiseless": bool(self.is_noiseless),
            "noise_application_scope": str(self.noise_application_scope),
            "parameters": self.parameter_metadata(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_readout_error(self, probability: float) -> float:
        error_probability = float(self.readout_error_probability)
        if error_probability <= 0.0:
            return float(probability)
        return float((1.0 - error_probability) * probability + error_probability * (1.0 - probability))

    def _coherent_noise_is_noiseless(self) -> bool:
        if self.noise_model_name != "coherent_overrotation":
            return self.coherent_overrotation_angle == 0.0
        if self.coherent_overrotation_mode == "stochastic" and self.coherent_overrotation_probability == 0.0:
            return True
        if self.coherent_overrotation_mode == "random_angle" and (
            self.coherent_overrotation_angle == 0.0 and self.coherent_overrotation_angle_std == 0.0
        ):
            return True
        if self.coherent_overrotation_mode == "pair_dependent":
            if self.pair_dependent_overrotation_angles is None:
                return self.coherent_overrotation_angle == 0.0
            return all(float(angle) == 0.0 for angle in self.pair_dependent_overrotation_angles)
        return self.coherent_overrotation_angle == 0.0


def noise_config_from_strength(
    noise_model_name: str,
    noise_strength: float,
    *,
    readout_error_probability: float = 0.0,
    coherent_overrotation_mode: str = "fixed",
    coherent_overrotation_probability: float = 1.0,
    coherent_overrotation_angle_std: float = 0.0,
    coherent_overrotation_seed: int | None = None,
    pair_dependent_overrotation_angles: tuple[float, ...] | None = None,
    noise_application_scope: str = "active",
    noisy_qubits: tuple[int, ...] | None = None,
    single_qubit_error_profile: tuple[float, ...] | None = None,
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
            noise_application_scope=noise_application_scope,
            noisy_qubits=noisy_qubits,
            single_qubit_error_profile=single_qubit_error_profile,
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "depolarizing":
        return NoiseConfig(
            noise_model_name="depolarizing",
            single_qubit_depolarizing_error=strength,
            two_qubit_depolarizing_error=strength,
            noise_application_scope=noise_application_scope,
            noisy_qubits=noisy_qubits,
            single_qubit_error_profile=single_qubit_error_profile,
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "amplitude_damping":
        return NoiseConfig(
            noise_model_name="amplitude_damping",
            amplitude_damping_gamma=strength,
            noise_application_scope=noise_application_scope,
            noisy_qubits=noisy_qubits,
            single_qubit_error_profile=single_qubit_error_profile,
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "phase_damping":
        return NoiseConfig(
            noise_model_name="phase_damping",
            phase_damping_gamma=strength,
            noise_application_scope=noise_application_scope,
            noisy_qubits=noisy_qubits,
            single_qubit_error_profile=single_qubit_error_profile,
            readout_error_probability=readout_error_probability,
        )
    if canonical_name == "coherent_overrotation":
        return NoiseConfig(
            noise_model_name="coherent_overrotation",
            coherent_overrotation_angle=strength,
            coherent_overrotation_mode=coherent_overrotation_mode,
            coherent_overrotation_probability=coherent_overrotation_probability,
            coherent_overrotation_angle_std=coherent_overrotation_angle_std,
            coherent_overrotation_seed=coherent_overrotation_seed,
            pair_dependent_overrotation_angles=pair_dependent_overrotation_angles,
            readout_error_probability=readout_error_probability,
            noise_application_scope=noise_application_scope,
            noisy_qubits=noisy_qubits,
            single_qubit_error_profile=single_qubit_error_profile,
        )
    raise ValueError(
        "noise_model_name must be one of "
        f"{SUPPORTED_NOISE_MODELS} (aliases: {tuple(_NOISE_MODEL_ALIASES)})"
    )
