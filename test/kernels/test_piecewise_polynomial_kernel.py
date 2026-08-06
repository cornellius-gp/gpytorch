#!/usr/bin/env python3

import math
import unittest

import torch

from gpytorch.kernels import PiecewisePolynomialKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestPiecewisePolynomialKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return PiecewisePolynomialKernel(q=2, **kwargs)

    def test_computes_piecewise_polynomial_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PiecewisePolynomialKernel(q=0)
        kernel.eval()

        def test_r(a, b):
            return torch.cdist(a, b)

        def test_get_cov(r, j, q):
            if q == 0:
                return 1
            if q == 1:
                return (j + 1) * r + 1
            if q == 2:
                return 1 + (j + 2) * r + ((j**2 + 4 * j + 3) / 3.0) * r**2
            if q == 3:
                return (
                    1
                    + (j + 3) * r
                    + ((6 * j**2 + 36 * j + 45) / 15.0) * r**2
                    + ((j**3 + 9 * j**2 + 23 * j + 15) / 15.0) * r**3
                )

        def test_fmax(r, j, q):
            return torch.max(torch.tensor(0.0), 1 - r).pow(j + q)

        actual = torch.zeros(3, 3)
        j = torch.floor(a / 2.0).shape[-1] + kernel.q + 1
        r = test_r(a, b)
        actual = test_fmax(r, j, kernel.q) * test_get_cov(r, j, kernel.q)
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        actual = actual.diagonal(dim1=-1, dim2=-2)
        res = kernel(a, b).diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 3, 3)
        for i in range(2):
            actual[i] = kernel(a[:, i].unsqueeze(-1), b[:, i].unsqueeze(-1)).to_dense()

        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=-1, dim2=-2)
        actual = torch.cat([actual[i].diagonal(dim1=-1, dim2=-2).unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_piecewise_polynomial_kernel_q2(self):
        # Points must be close enough that r < 1 (i.e. inside the kernel's
        # compact support), otherwise (1 - r)_+ zeros out the polynomial term
        # and the q=2 coefficient is never exercised.
        a = torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.3, 0.1]], dtype=torch.float)
        b = torch.tensor([[0.05, 0.05], [0.2, 0.1], [0.15, 0.25]], dtype=torch.float)
        D = a.shape[-1]
        kernel = PiecewisePolynomialKernel(q=2)
        kernel.eval()

        # Reconstruct the reference value using the kernel's own distance (so the
        # only thing under test is the q=2 covariance polynomial coefficient).
        r = kernel.covar_dist(a.div(kernel.lengthscale), b.div(kernel.lengthscale))
        self.assertTrue((r < 1).any())  # ensure the polynomial term is actually exercised
        j = math.floor(D / 2.0) + kernel.q + 1
        fmax = torch.clamp(1 - r, min=0.0).pow(j + kernel.q)
        # Closed form from Rasmussen & Williams Eq. 4.21
        cov = 1 + (j + 2) * r + ((j**2 + 4 * j + 3) / 3.0) * r**2
        actual = fmax * cov
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_piecewise_polynomial_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [-1, 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PiecewisePolynomialKernel(q=0, batch_shape=torch.Size([2]))
        kernel.eval()

        def test_r(a, b):
            return torch.cdist(a, b)

        def test_get_cov(r, j, q):
            if q == 0:
                return 1
            if q == 1:
                return (j + 1) * r + 1
            if q == 2:
                return 1 + (j + 2) * r + ((j**2 + 4 * j + 3) / 3.0) * r**2
            if q == 3:
                return (
                    1
                    + (j + 3) * r
                    + ((6 * j**2 + 36 * j + 45) / 15.0) * r**2
                    + ((j**3 + 9 * j**2 + 23 * j + 15) / 15.0) * r**3
                )

        def test_fmax(r, j, q):
            return torch.max(torch.tensor(0.0), 1 - r).pow(j + q)

        actual = torch.zeros(3, 3)
        j = torch.floor(a / 2.0).shape[-1] + kernel.q + 1
        r = test_r(a, b)
        actual = test_fmax(r, j, kernel.q) * test_get_cov(r, j, kernel.q)
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
