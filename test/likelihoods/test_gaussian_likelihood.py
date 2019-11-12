#!/usr/bin/env python3

import unittest

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood()


class TestGaussianLikelihoodBatch(TestGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood(batch_shape=torch.Size([3]))

    def test_nonbatch(self):
        pass


class TestGaussianLikelihoodMultiBatch(TestGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood(batch_shape=torch.Size([2, 3]))

    def test_nonbatch(self):
        pass

    def test_batch(self):
        pass


class TestFixedNoiseGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_fixed_noise_gaussian_likelihood(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            lkhd = FixedNoiseGaussianLikelihood(noise=noise)
            # test basics
            self.assertIsInstance(lkhd.noise_covar, FixedGaussianNoise)
            self.assertTrue(torch.equal(noise, lkhd.noise))
            new_noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            lkhd.noise = new_noise
            self.assertTrue(torch.equal(lkhd.noise, new_noise))
            # test __call__
            mean = torch.zeros(4, device=device, dtype=dtype)
            covar = DiagLazyTensor(torch.ones(4, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            out = lkhd(mvn)
            self.assertTrue(torch.allclose(out.variance, 1 + new_noise))
            # things should break if dimensions mismatch
            mean = torch.zeros(5, device=device, dtype=dtype)
            covar = DiagLazyTensor(torch.ones(5, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            with self.assertWarns(UserWarning):
                lkhd(mvn)
            # test __call__ w/ observation noise
            obs_noise = 0.1 + torch.rand(5, device=device, dtype=dtype)
            out = lkhd(mvn, noise=obs_noise)
            self.assertTrue(torch.allclose(out.variance, 1 + obs_noise))


class TestFixedNoiseGaussianLikelihoodBatch(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(3, 5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_nonbatch(self):
        pass


class TestFixedNoiseGaussianLikelihoodMultiBatch(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(2, 3, 5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_nonbatch(self):
        pass

    def test_batch(self):
        pass


if __name__ == "__main__":
    unittest.main()
