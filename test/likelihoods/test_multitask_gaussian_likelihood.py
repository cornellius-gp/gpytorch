#!/usr/bin/env python3

import unittest

import torch
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator, ToeplitzLinearOperator

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import _MultitaskGaussianLikelihoodBase, MultitaskGaussianLikelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.priors import LKJPrior
from gpytorch.priors.wishart_prior import WishartPrior
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestMultitaskGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 2

    def _create_conditional_input(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        data_mat = ToeplitzLinearOperator(torch.tensor([1, 0.6, 0.4, 0.2, 0.1]))
        task_mat = RootLinearOperator(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        covar = KroneckerProductLinearOperator(data_mat, task_mat)
        return MultitaskMultivariateNormal(torch.randn(*batch_shape, 5, 4), covar)

    def _create_targets(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def create_likelihood(self):
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2)

    def test_marginal_variance(self):
        likelihood = MultitaskGaussianLikelihood(num_tasks=4, rank=0, has_global_noise=False)
        likelihood.task_noises = torch.tensor([[0.1], [0.2], [0.3], [0.4]])

        input = self._create_marginal_input()
        variance = likelihood(input).variance
        self.assertAllClose(variance, torch.tensor([1.1, 4.2, 9.3, 16.4]).repeat(5, 1))

        likelihood = MultitaskGaussianLikelihood(num_tasks=4, rank=1, has_global_noise=True)
        likelihood.noise = torch.tensor(0.1)
        likelihood.task_noise_covar_factor.data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

        input = self._create_marginal_input()
        variance = likelihood(input).variance
        self.assertAllClose(variance, torch.tensor([2.1, 8.1, 18.1, 32.1]).repeat(5, 1))

    def test_setters(self):
        likelihood = MultitaskGaussianLikelihood(num_tasks=3, rank=0)

        a = torch.randn(3, 2)
        mat = a.matmul(a.transpose(-1, -2))

        # test rank 0 setters
        likelihood.noise = 0.5
        self.assertAlmostEqual(0.5, likelihood.noise.item())

        likelihood.task_noises = torch.tensor([0.04, 0.04, 0.04])
        for i in range(3):
            self.assertAlmostEqual(0.04, likelihood.task_noises[i].item())

        with self.assertRaises(AttributeError) as context:
            likelihood.task_noise_covar = mat
        self.assertTrue("task noises" in str(context.exception))

        # test low rank setters
        likelihood = MultitaskGaussianLikelihood(num_tasks=3, rank=2)
        likelihood.noise = 0.5
        self.assertAlmostEqual(0.5, likelihood.noise.item())

        likelihood.task_noise_covar = mat
        self.assertAllClose(mat, likelihood.task_noise_covar)

        with self.assertRaises(AttributeError) as context:
            likelihood.task_noises = torch.tensor([0.04, 0.04, 0.04])
        self.assertTrue("task noises" in str(context.exception))


class TestMultitaskGaussianLikelihoodNonInterleaved(TestMultitaskGaussianLikelihood, unittest.TestCase):
    seed = 2

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        data_mat = ToeplitzLinearOperator(torch.tensor([1, 0.6, 0.4, 0.2, 0.1]))
        task_mat = RootLinearOperator(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        covar = KroneckerProductLinearOperator(task_mat, data_mat)
        return MultitaskMultivariateNormal(torch.randn(*batch_shape, 5, 4), covar, interleaved=False)


class TestMultitaskGaussianLikelihoodBatch(TestMultitaskGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2, batch_shape=torch.Size([3]))

    def test_nonbatch(self):
        pass

    def test_batched_likelihood_with_unbatched_and_broadcast_inputs(self):
        unbatched_mvn = self._create_marginal_input(batch_shape=torch.Size([]))
        broadcast_mvn = self._create_marginal_input(batch_shape=torch.Size([2, 1]))
        unbatched_samples = self._create_conditional_input(batch_shape=torch.Size([]))
        for has_task_noise, has_global_noise, rank in [
            (True, True, 0),
            (True, False, 0),
            (True, True, 2),
            (True, False, 2),
            (False, True, 0),
        ]:
            lik = MultitaskGaussianLikelihood(
                num_tasks=4,
                rank=rank,
                batch_shape=torch.Size([3]),
                has_task_noise=has_task_noise,
                has_global_noise=has_global_noise,
            )
            out_unbatched = lik(unbatched_mvn)
            self.assertEqual(out_unbatched.batch_shape, torch.Size([3]))
            self.assertEqual(out_unbatched.event_shape, torch.Size([5, 4]))

            out_broadcast = lik(broadcast_mvn)
            self.assertEqual(out_broadcast.batch_shape, torch.Size([2, 3]))
            self.assertEqual(out_broadcast.event_shape, torch.Size([5, 4]))

            cond_unbatched = lik(unbatched_samples)
            self.assertEqual(cond_unbatched.batch_shape, torch.Size([3, 5]))
            self.assertEqual(cond_unbatched.event_shape, torch.Size([4]))

    def test_task_prior_and_correlation_prior_closures(self):
        covar_prior = WishartPrior(nu=6, K=torch.eye(4))
        lik = MultitaskGaussianLikelihood(
            num_tasks=4,
            rank=2,
            batch_shape=torch.Size([3]),
            task_prior=covar_prior,
            has_global_noise=True,
        )
        for _, module, prior, closure, _ in lik.named_priors():
            val = closure(module)
            self.assertEqual(val.shape, torch.Size([3, 4, 4]))
            lp = prior.log_prob(val)
            self.assertEqual(lp.shape, torch.Size([3]))
            self.assertTrue(torch.isfinite(lp).all())

        lik_no_global = MultitaskGaussianLikelihood(
            num_tasks=4,
            rank=4,
            task_prior=WishartPrior(nu=6, K=torch.eye(4)),
            has_global_noise=False,
        )
        lik_no_global.task_noise_covar_factor.data.copy_(torch.eye(4))
        for _, module, prior, closure, _ in lik_no_global.named_priors():
            val = closure(module)
            self.assertEqual(val.shape, torch.Size([4, 4]))
            lp = prior.log_prob(val)
            self.assertTrue(torch.isfinite(lp).all())

        base_lik = _MultitaskGaussianLikelihoodBase(
            num_tasks=4,
            noise_covar=HomoskedasticNoise(),
            rank=2,
            task_correlation_prior=LKJPrior(n=4, eta=1.0),
            batch_shape=torch.Size([3]),
        )
        found_corr_prior = False
        for name, module, prior, closure, _ in base_lik.named_priors():
            if name == "MultitaskErrorCorrelationPrior":
                found_corr_prior = True
                corr = closure(module)
                self.assertEqual(corr.shape, torch.Size([3, 4, 4]))
                lp = prior.log_prob(corr)
                self.assertEqual(lp.shape, torch.Size([3]))
                self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue(found_corr_prior)
