#!/usr/bin/env python3

import unittest

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from gpytorch.likelihoods import (
    DirichletClassificationLikelihood,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    GaussianLikelihoodWithMissingObs,
)
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


class TestDirichletClassificationLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        train_x = torch.randn(15)
        labels = torch.round(train_x).long()
        likelihood = DirichletClassificationLikelihood(labels)
        return likelihood

    def test_batch(self):
        pass

    def test_multi_batch(self):
        pass

    def test_nonbatch(self):
        pass

    def test_dirichlet_classification_likelihood(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            noise = torch.rand(6, device=device, dtype=dtype) > 0.5
            noise = noise.long()
            lkhd = DirichletClassificationLikelihood(noise, dtype=dtype)
            # test basics
            self.assertIsInstance(lkhd.noise_covar, FixedGaussianNoise)
            noise = torch.rand(6, device=device, dtype=dtype) > 0.5
            noise = noise.long()
            new_noise, _, _ = lkhd._prepare_targets(noise, dtype=dtype)
            lkhd.noise = new_noise
            self.assertTrue(torch.equal(lkhd.noise, new_noise))
            # test __call__
            mean = torch.zeros(6, device=device, dtype=dtype)
            covar = DiagLazyTensor(torch.ones(6, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            out = lkhd(mvn)
            self.assertTrue(torch.allclose(out.variance, 1 + new_noise))
            # things should break if dimensions mismatch
            mean = torch.zeros(5, device=device, dtype=dtype)
            covar = DiagLazyTensor(torch.ones(5, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            with self.assertWarns(UserWarning):
                lkhd(mvn)
            # test __call__ w/ new targets
            obs_noise = 0.1 + torch.rand(5, device=device, dtype=dtype)
            obs_noise = (obs_noise > 0.5).long()
            out = lkhd(mvn, targets=obs_noise)
            obs_targets, _, _ = lkhd._prepare_targets(obs_noise, dtype=dtype)
            self.assertTrue(torch.allclose(out.variance, 1.0 + obs_targets))


class TestGaussianLikelihoodwithMissingObs(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 42

    def create_likelihood(self):
        return GaussianLikelihoodWithMissingObs()

    def test_missing_value_inference(self):
        """
        samples = mvn samples + noise samples
        In this test, we try to recover noise parameters when some elements in
        'samples' are missing at random.
        """

        torch.manual_seed(self.seed)

        mu = torch.zeros(2, 3)
        sigma = torch.tensor([[[1, 0.999, -0.999], [0.999, 1, -0.999], [-0.999, -0.999, 1]]] * 2).float()
        mvn = MultivariateNormal(mu, sigma)
        samples = mvn.sample(torch.Size([10000]))  # mvn samples

        noise_sd = 0.5
        noise_dist = torch.distributions.Normal(0, noise_sd)
        samples += noise_dist.sample(samples.shape)  # noise

        missing_prop = 0.33
        missing_idx = torch.distributions.Binomial(1, missing_prop).sample(samples.shape).bool()
        samples[missing_idx] = float("nan")

        likelihood = GaussianLikelihoodWithMissingObs()

        # check that the missing value fill doesn't impact the likelihood

        likelihood.MISSING_VALUE_FILL = 999.0
        like_init_plus = likelihood.log_marginal(samples, mvn).sum().data

        likelihood.MISSING_VALUE_FILL = -999.0
        like_init_minus = likelihood.log_marginal(samples, mvn).sum().data

        torch.testing.assert_allclose(like_init_plus, like_init_minus)

        # check that the correct noise sd is recovered

        opt = torch.optim.Adam(likelihood.parameters(), lr=0.05)

        for _ in range(100):
            opt.zero_grad()
            loss = -likelihood.log_marginal(samples, mvn).sum()
            loss.backward()
            opt.step()

        assert abs(float(likelihood.noise.sqrt()) - 0.5) < 0.02

        # Check log marginal works
        likelihood.log_marginal(samples[0], mvn)


if __name__ == "__main__":
    unittest.main()
