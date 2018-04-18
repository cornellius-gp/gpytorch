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

    def test_solve_vector(self):
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

    def test_solve(self):
        size = 100
        train_x = Variable(torch.linspace(0, 1, size))
        covar_matrix = RBFKernel()(train_x, train_x).data
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        woodbury_factor = pivoted_cholesky.woodbury_factor(piv_chol, 1)

        rhs_vector = torch.randn(100, 50)
        shifted_covar_matrix = covar_matrix + torch.eye(size)
        real_solve = shifted_covar_matrix.inverse().matmul(rhs_vector)
        approx_solve = pivoted_cholesky.woodbury_solve(rhs_vector, piv_chol, woodbury_factor, 1)

        self.assertTrue(approx_equal(approx_solve, real_solve))


class TestPivotedCholeskyBatch(unittest.TestCase):
    def test_pivoted_cholesky(self):
        size = 100
        train_x = Variable(torch.cat([
            torch.linspace(0, 1, size).unsqueeze(0),
            torch.linspace(0, 0.5, size).unsqueeze(0),
        ], 0)).unsqueeze(-1)
        covar_matrix = RBFKernel()(train_x, train_x).data
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol.transpose(1, 2).matmul(piv_chol)

        self.assertTrue(approx_equal(covar_approx, covar_matrix))

    def test_solve(self):
        size = 100
        train_x = Variable(torch.cat([
            torch.linspace(0, 1, size).unsqueeze(0),
            torch.linspace(0, 0.5, size).unsqueeze(0),
        ], 0)).unsqueeze(-1)
        covar_matrix = RBFKernel()(train_x, train_x).data
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        woodbury_factor = pivoted_cholesky.woodbury_factor(piv_chol, 1)

        rhs_vector = torch.randn(2, 100, 5)
        shifted_covar_matrix = covar_matrix + torch.eye(size)
        real_solve = torch.cat([
            shifted_covar_matrix[0].inverse().matmul(rhs_vector[0]).unsqueeze(0),
            shifted_covar_matrix[1].inverse().matmul(rhs_vector[1]).unsqueeze(0),
        ], 0)
        approx_solve = pivoted_cholesky.woodbury_solve(rhs_vector, piv_chol, woodbury_factor, 1)

        self.assertTrue(approx_equal(approx_solve, real_solve))


if __name__ == '__main__':
    unittest.main()
