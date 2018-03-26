from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.utils import pivoted_cholesky, approx_equal
from gpytorch.kernels import RBFKernel


class TestPivotedCholesky(unittest.TestCase):

    def test_pivoted_cholesky(self):
        size = 100
        train_x = Variable(torch.linspace(0, 1, size))
        covar_matrix = RBFKernel()(train_x, train_x).data
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol.t().matmul(piv_chol)

        self.assertTrue(approx_equal(covar_approx, covar_matrix))

    def test_solve(self):
        size = 100
        train_x = Variable(torch.linspace(0, 1, size))
        covar_matrix = RBFKernel()(train_x, train_x).data
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        woodbury_factor = pivoted_cholesky.woodbury_factor(piv_chol, 1)

        rhs_vector = torch.randn(100)
        shifted_covar_matrix = covar_matrix + torch.eye(size)
        real_solve = shifted_covar_matrix.inverse().matmul(rhs_vector)
        approx_solve = pivoted_cholesky.woodbury_solve(rhs_vector, piv_chol, woodbury_factor, 1)

        self.assertTrue(approx_equal(approx_solve, real_solve))


if __name__ == '__main__':
    unittest.main()
