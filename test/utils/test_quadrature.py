#!/usr/bin/env python3

import os
import random
import unittest

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D


class TestQuadrature(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_gauss_hermite_quadrature_1D_normal_nonbatch(self, cuda=False):
        func = lambda x: torch.sin(x)

        means = torch.randn(10)
        variances = torch.randn(10).abs()
        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = torch.distributions.Normal(means, variances.sqrt())

        # Use quadrature
        results = quadrature(func, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = func(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_normal_nonbatch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)

    def test_gauss_hermite_quadrature_1D_normal_batch(self, cuda=False):
        func = lambda x: torch.sin(x)

        means = torch.randn(3, 10)
        variances = torch.randn(3, 10).abs()
        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = torch.distributions.Normal(means, variances.sqrt())

        # Use quadrature
        results = quadrature(func, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = func(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_normal_batch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)

    def test_gauss_hermite_quadrature_1D_mvn_nonbatch(self, cuda=False):
        func = lambda x: torch.sin(x)

        means = torch.randn(10)
        variances = torch.randn(10).abs()

        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = MultivariateNormal(means, DiagLazyTensor(variances.sqrt()))

        # Use quadrature
        results = quadrature(func, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = func(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_mvn_nonbatch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)

    def test_gauss_hermite_quadrature_1D_mvn_batch(self, cuda=False):
        func = lambda x: torch.sin(x)

        means = torch.randn(3, 10)
        variances = torch.randn(3, 10).abs()
        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = MultivariateNormal(means, DiagLazyTensor(variances.sqrt()))

        # Use quadrature
        results = quadrature(func, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = func(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_mvn_batch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)


if __name__ == "__main__":
    unittest.main()
