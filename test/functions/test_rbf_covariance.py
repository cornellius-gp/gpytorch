#!/usr/bin/env python3

import unittest

import torch

import gpytorch


def sq_dist_func(x1, x2):
    dist_module = gpytorch.kernels.kernel.Distance()
    return dist_module._sq_dist(x1, x2, postprocess=torch.tensor(False))


class TestRBFCovariance(unittest.TestCase):
    def test_forward(self):
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2
        res = gpytorch.functions.RBFCovariance.apply(x1, x2, lengthscale, sq_dist_func)
        actual = sq_dist_func(x1, x2).div(-2 * lengthscale ** 2).exp()
        self.assertTrue(torch.allclose(res, actual))

    def test_backward(self):
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.RBFCovariance.apply(x1, x2, l, sq_dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")


if __name__ == "__main__":
    unittest.main()
