"""Physics utilities for EQNN experiments."""

from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.physics.observables import (
    SINGLET_PROJECTOR,
    SINGLET_STATE,
    SWAP_OPERATOR,
    alternating_singlet_means,
    dimerization_feature,
    singlet_fraction,
    swap_expectation,
    swap_probability,
)
from eqnn.physics.quantum import (
    as_density_matrix,
    embed_operator_on_sites,
    expectation_value,
    partial_trace_density_matrix,
    partial_trace_adjoint,
    qubit_permutation_operator,
    reduced_density_matrix,
    statevector_to_density_matrix,
)

__all__ = [
    "BondAlternatingHeisenbergHamiltonian",
    "SINGLET_PROJECTOR",
    "SINGLET_STATE",
    "SWAP_OPERATOR",
    "alternating_singlet_means",
    "as_density_matrix",
    "embed_operator_on_sites",
    "dimerization_feature",
    "expectation_value",
    "partial_trace_adjoint",
    "partial_trace_density_matrix",
    "qubit_permutation_operator",
    "reduced_density_matrix",
    "singlet_fraction",
    "statevector_to_density_matrix",
    "swap_expectation",
    "swap_probability",
]
