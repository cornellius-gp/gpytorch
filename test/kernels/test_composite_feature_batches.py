#!/usr/bin/env python3

import itertools
import unittest

import torch

from gpytorch import settings
from gpytorch.kernels import AdditiveKernel, ProductKernel, RBFKernel, ScaleKernel


class TestCompositeFeatureBatches(unittest.TestCase):
    def test_selected_feature_batches(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, product, scaled, broadcast in itertools.product(
            devices, (False, True), (False, True), (False, True)
        ):
            with self.subTest(device=device, product=product, scaled=scaled, broadcast=broadcast):
                left = RBFKernel(active_dims=[2, 0])
                right = RBFKernel(active_dims=[1] if broadcast else [1, 2])
                kernel = (ProductKernel if product else AdditiveKernel)(left, right)
                if scaled:
                    kernel = ScaleKernel(kernel)
                kernel = kernel.to(device=device, dtype=torch.double)
                X = torch.rand(4, 3, device=device, dtype=torch.double, requires_grad=True)
                Y = torch.rand(5, 3, device=device, dtype=torch.double)
                lazy = kernel(X, Y, last_dim_is_batch=True)
                self.assertEqual(lazy.shape, torch.Size([2, 4, 5]))
                with settings.lazily_evaluate_kernels(False):
                    expected = kernel(X, Y, last_dim_is_batch=True).to_dense()
                actual = lazy.to_dense()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(lazy[1].to_dense(), expected[1])
                actual_grad = torch.autograd.grad(actual.sum(), X, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected.sum(), X)[0]
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_scale_selects_active_dimensions_once(self):
        base = RBFKernel(active_dims=[2, 0])
        kernel = ScaleKernel(base) + RBFKernel(active_dims=[1, 2])
        X = torch.rand(4, 3)
        lazy = kernel(X, last_dim_is_batch=True)
        with settings.lazily_evaluate_kernels(False):
            expected = kernel(X, last_dim_is_batch=True).to_dense()
        self.assertEqual(lazy.shape, torch.Size([2, 4, 4]))
        torch.testing.assert_close(lazy.to_dense(), expected)
        torch.testing.assert_close(lazy.diagonal(), expected.diagonal(dim1=-2, dim2=-1))
