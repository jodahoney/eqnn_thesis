"""Circuit-construction helpers for backend integrations."""

from eqnn.circuits.qiskit_builders import (
    QISKIT_AVAILABLE,
    build_anisotropic_pair_block,
    build_convolution_circuit,
    build_hea_pair_block,
    build_state_preparation_circuit,
    build_su2_pair_block,
    build_swap_readout_operator,
    qiskit_density_to_repo,
    repo_density_to_qiskit,
    repo_sites_to_qiskit,
)

__all__ = [
    "QISKIT_AVAILABLE",
    "build_anisotropic_pair_block",
    "build_convolution_circuit",
    "build_hea_pair_block",
    "build_state_preparation_circuit",
    "build_su2_pair_block",
    "build_swap_readout_operator",
    "qiskit_density_to_repo",
    "repo_density_to_qiskit",
    "repo_sites_to_qiskit",
]
