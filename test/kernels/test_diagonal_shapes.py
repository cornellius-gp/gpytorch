#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import IndexKernel, RBFKernel


class TestDiagonalShapes(unittest.TestCase):
    def test_feature_and_point_counts_equal(self):
        for batch in [(), (2,)]:
            with self.subTest(batch=batch):
                kernel = RBFKernel().double()
                x1 = torch.rand(*batch, 3, 3, dtype=torch.double)
                x2 = torch.rand(*batch, 3, 3, dtype=torch.double)
                expected = kernel(x1, x2, last_dim_is_batch=True).to_dense().diagonal(dim1=-1, dim2=-2)
                actual = kernel(x1, x2, diag=True, last_dim_is_batch=True)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.allclose(actual, expected))

    def test_ignored_diagonal_with_model_batch(self):
        kernel = IndexKernel(num_tasks=3, batch_shape=torch.Size([2]))
        x = torch.arange(3).unsqueeze(-1)
        actual = kernel(x, diag=True)
        expected = kernel(x).to_dense().diagonal(dim1=-1, dim2=-2)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected))
