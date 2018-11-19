#!/usr/bin/env python3

import torch
import unittest
from test._utils import approx_equal
from gpytorch.utils.lanczos import lanczos_tridiag


class TestLanczos(unittest.TestCase):
    def test_lanczos(self):
        size = 100
        matrix = torch.randn(size, size)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.ones(matrix.size(-1)).mul(1e-6).diag())
        q_mat, t_mat = lanczos_tridiag(
            matrix.matmul, max_iter=size, dtype=matrix.dtype, device=matrix.device, matrix_shape=matrix.shape
        )

        approx = q_mat.matmul(t_mat).matmul(q_mat.transpose(-1, -2))
        self.assertTrue(approx_equal(approx, matrix))


if __name__ == "__main__":
    unittest.main()
