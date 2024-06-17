#!/usr/bin/env python3

import unittest

from gpytorch.models.computation_aware_iterative_gp import preconditioners
from .test_preconditioner import BasePreconditionerTestCase

N_PTS = 100


class TestDiagonal(unittest.TestCase, BasePreconditionerTestCase):
    """Tests for a diagonal preconditioner."""

    def create_preconditioner(self) -> preconditioners.Preconditioner:
        return preconditioners.Diagonal(
            kernel=self.create_kernel(), noise=self.create_likelihood().noise, X=self.create_train_data()
        )


if __name__ == "__main__":
    unittest.main()
