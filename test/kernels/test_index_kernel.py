#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import IndexKernel, RBFKernel
from gpytorch.priors import NormalPrior


class TestIndexKernel(unittest.TestCase):
    def create_kernel_with_prior(self, prior):
        return IndexKernel(num_tasks=1, prior=prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        kernel_fn = lambda prior: self.create_kernel_with_prior(prior)
        kernel_fn(None)
        kernel_fn(NormalPrior(0, 1))
        self.assertRaises(TypeError, kernel_fn, 1)

    def test_diag(self):
        # diag=True returns the diagonal of the covariance matrix, also for batched kernels or
        # inputs, in the lazily evaluated kernel tensor and in products with other kernels.
        i1 = torch.tensor([[0], [1], [2], [1]])
        i2 = torch.tensor([[2], [1], [0], [0]])
        x = torch.rand(4, 1)
        for batch_shape, input_batch_shape in (
            (torch.Size([]), torch.Size([])),
            (torch.Size([2]), torch.Size([])),
            (torch.Size([]), torch.Size([3])),
            (torch.Size([2]), torch.Size([2])),
        ):
            kernel = IndexKernel(num_tasks=3, rank=2, batch_shape=batch_shape)
            kernel.covar_factor.data.normal_()
            kernel.var = torch.rand(*batch_shape, 3) + 0.1
            expected_shape = torch.broadcast_shapes(batch_shape, input_batch_shape) + torch.Size([4])
            for j1, j2 in ((i1, i1), (i1, i2)):
                j1 = j1.expand(*input_batch_shape, 4, 1)
                j2 = j2.expand(*input_batch_shape, 4, 1)
                expected = kernel(j1, j2).to_dense().diagonal(dim1=-2, dim2=-1)
                self.assertEqual(expected.shape, expected_shape)
                res = kernel(j1, j2, diag=True)
                self.assertEqual(res.shape, expected_shape)
                self.assertTrue(torch.allclose(res, expected))
                self.assertTrue(torch.allclose(kernel(j1, j2).diagonal(dim1=-2, dim2=-1), expected))
            # a product kernel, as in multi-task GPs
            product = RBFKernel(batch_shape=batch_shape, active_dims=[0]) * IndexKernel(
                num_tasks=3, rank=2, batch_shape=batch_shape, active_dims=[1]
            )
            xi = torch.cat([x, i1], dim=-1).expand(*input_batch_shape, 4, 2)
            expected = product(xi).to_dense().diagonal(dim1=-2, dim2=-1)
            self.assertTrue(torch.allclose(product(xi, diag=True), expected))
