#!/usr/bin/env python3

import unittest

from gpytorch.models.computation_aware_iterative_gp import preconditioners
from .test_preconditioner import BasePreconditionerTestCase

N_PTS = 100


class TestSparseInverseCholesky(unittest.TestCase, BasePreconditionerTestCase):
    """Tests for a diagonal preconditioner."""

    def create_preconditioner(self) -> preconditioners.Preconditioner:
        return preconditioners.SparseInverseCholesky(
            kernel=self.create_kernel(), noise=self.create_likelihood().noise, X=self.create_train_data()
        )

    def test_sparsity_set_is_whole_dataset_gives_inverse():
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
