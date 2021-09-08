#!/usr/bin/env python3

import math
import unittest

import torch

import gpytorch


def dist_func(x1, x2):
    dist_module = gpytorch.kernels.kernel.Distance()
    return dist_module._dist(x1, x2, postprocess=torch.tensor(False))


class TestMaternCovariance(unittest.TestCase):
    def test_1_2_forward(self):
        nu = 1 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2

        res = gpytorch.functions.MaternCovariance.apply(x1, x2, lengthscale, nu, dist_func)
        scaled_unitless_dist = math.sqrt(nu * 2) * dist_func(x1, x2).div(lengthscale)
        exp_component = torch.exp(-scaled_unitless_dist)
        actual = exp_component
        self.assertTrue(torch.allclose(res, actual))

    def test_1_2_backward(self):
        nu = 1 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.MaternCovariance.apply(x1, x2, l, nu, dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")

    def test_3_2_forward(self):
        nu = 3 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2

        res = gpytorch.functions.MaternCovariance.apply(x1, x2, lengthscale, nu, dist_func)
        scaled_unitless_dist = math.sqrt(nu * 2) * dist_func(x1, x2).div(lengthscale)
        exp_component = torch.exp(-scaled_unitless_dist)
        actual = exp_component * (1 + scaled_unitless_dist)
        self.assertTrue(torch.allclose(res, actual))

    def test_3_2_backward(self):
        nu = 3 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.MaternCovariance.apply(x1, x2, l, nu, dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")

    def test_5_2_forward(self):
        nu = 5 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2

        res = gpytorch.functions.MaternCovariance.apply(x1, x2, lengthscale, nu, dist_func)
        scaled_unitless_dist = math.sqrt(nu * 2) * dist_func(x1, x2).div(lengthscale)
        exp_component = torch.exp(-scaled_unitless_dist)
        actual = exp_component * (1 + scaled_unitless_dist + scaled_unitless_dist ** 2 / 3)
        self.assertTrue(torch.allclose(res, actual))

    def test_5_2_backward(self):
        nu = 5 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.MaternCovariance.apply(x1, x2, l, nu, dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")


if __name__ == "__main__":
    unittest.main()
