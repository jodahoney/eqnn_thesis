from __future__ import annotations

import unittest

import numpy as np

from eqnn.physics.observables import SINGLET_STATE, swap_expectation, swap_probability
from eqnn.physics.quantum import as_density_matrix


class ObservableTests(unittest.TestCase):
    def test_swap_observable_identifies_singlet_and_triplet_sectors(self) -> None:
        singlet_density = as_density_matrix(SINGLET_STATE)
        triplet_density = as_density_matrix(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128))

        self.assertAlmostEqual(swap_expectation(singlet_density), -1.0, places=12)
        self.assertAlmostEqual(swap_probability(singlet_density), 0.0, places=12)
        self.assertAlmostEqual(swap_expectation(triplet_density), 1.0, places=12)
        self.assertAlmostEqual(swap_probability(triplet_density), 1.0, places=12)

    def test_swap_observable_rejects_wrong_hilbert_space_dimension(self) -> None:
        with self.assertRaises(ValueError):
            swap_expectation(np.eye(8, dtype=np.complex128) / 8.0)
