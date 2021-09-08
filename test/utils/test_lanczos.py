#!/usr/bin/env python3

import unittest

import torch

from gpytorch.test.utils import approx_equal
from gpytorch.utils.lanczos import lanczos_tridiag
from scipy.stats import ortho_group


class TestLanczos(unittest.TestCase):
    def assert_valid_sizes(self, size, t_mat, q_mat):
        rank = t_mat.shape[0]
        self.assertTrue(0 < rank <= size)
        self.assertEqual(rank, t_mat.shape[1])
        self.assertEqual(rank, q_mat.shape[1])
        self.assertEqual(size, q_mat.shape[0])

    def assert_tridiagonally_positive(self, t_mat):
        for i in range(t_mat.shape[0]):
            for elem in t_mat.data[i, i - 1 : i + 2]:
                self.assertGreater(elem, 0)

    def lanczos_tridiag_test(self, matrix):
        size = matrix.shape[0]
        q_mat, t_mat = lanczos_tridiag(
            matrix.matmul, max_iter=size, dtype=matrix.dtype, device=matrix.device, matrix_shape=matrix.shape
        )

        self.assert_valid_sizes(size, t_mat, q_mat)
        self.assert_tridiagonally_positive(t_mat)
        approx = q_mat.matmul(t_mat).matmul(q_mat.transpose(-1, -2))
        self.assertTrue(approx_equal(approx, matrix))

    # this type of matrix has eigenvalues of similar scale, so our approximation will likely create a tridiaganal
    # matrix of the same size
    def test_lanczos_tridiag_near_exact(self):
        size = 100
        matrix = torch.randn(size, size)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.ones(matrix.size(-1)).mul(1e-6).diag())
        self.lanczos_tridiag_test(matrix)

    # this kind of matrix has eigenvalues on very different scales, so our approximation will likely create a
    # tridiagonal matrix of smaller size
    def test_lanczos_tridiag_approx(self):
        size = 30
        orthogonal = torch.from_numpy(ortho_group.rvs(size)).float()
        diag = torch.diag_embed(torch.FloatTensor([10 ** -i for i in range(size)]))
        matrix = torch.matmul(orthogonal, torch.matmul(diag, orthogonal.transpose(0, 1)))
        self.lanczos_tridiag_test(matrix)
