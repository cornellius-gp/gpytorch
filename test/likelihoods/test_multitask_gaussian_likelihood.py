#!/usr/bin/env python3

import unittest

import torch
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator, ToeplitzLinearOperator

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestMultitaskGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 2

    def _create_conditional_input(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        data_mat = ToeplitzLinearOperator(torch.tensor([1, 0.6, 0.4, 0.2, 0.1]))
        task_mat = RootLinearOperator(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        covar = KroneckerProductLinearOperator(data_mat, task_mat)
        covar = covar.expand(*batch_shape, *covar.matrix_shape)
        return MultitaskMultivariateNormal(torch.randn(*batch_shape, 5, 4), covar)

    def _create_targets(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def create_likelihood(self):
        # The generic Monte Carlo reference must retain latent task correlations.
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2).eval()

    def test_correlated_task_noise_log_probabilities(self):
        mean = torch.tensor(
            [[[0.2, -0.1], [0.4, 0.7]], [[-0.3, 0.5], [0.1, -0.2]]], dtype=torch.double, requires_grad=True
        )
        target = torch.tensor([[[0.1, 0.3], [0.2, -0.4]], [[0.5, 0.8], [-0.1, 0.4]]], dtype=torch.double)
        root = torch.tensor(
            [[0.4, 0.0, 0.0, 0.0], [0.1, 0.3, 0.0, 0.0], [0.1, 0.1, 0.4, 0.0], [0.0, 0.1, 0.2, 0.3]],
            dtype=torch.double,
            requires_grad=True,
        )
        latent_covar = root @ root.transpose(-1, -2) + 0.05 * torch.eye(4, dtype=torch.double)
        point_covars = torch.stack([latent_covar[:2, :2], latent_covar[2:, 2:]])
        likelihood = MultitaskGaussianLikelihood(num_tasks=2, rank=1, batch_shape=torch.Size([2])).double()
        with torch.no_grad():
            likelihood.task_noise_covar_factor.copy_(torch.tensor([[[0.7], [0.3]], [[-0.2], [0.6]]]))
        likelihood.noise = torch.tensor([[0.2], [0.4]], dtype=torch.double)
        factor = likelihood.task_noise_covar_factor
        noise_covar = factor @ factor.transpose(-1, -2) + likelihood.noise.unsqueeze(-1) * torch.eye(
            2, dtype=torch.double
        )
        conditional = torch.distributions.MultivariateNormal(mean, noise_covar.unsqueeze(-3))
        conditional_log_prob = conditional.log_prob(target)
        trace = torch.linalg.solve(noise_covar.unsqueeze(-3), point_covars).diagonal(dim1=-1, dim2=-2).sum(-1)
        references = {
            "forward": conditional_log_prob,
            "expected_log_prob": conditional_log_prob - 0.5 * trace,
            "log_marginal": torch.distributions.MultivariateNormal(
                mean, noise_covar.unsqueeze(-3) + point_covars
            ).log_prob(target),
        }
        parameters = (mean, root, factor, likelihood.raw_noise)
        for interleaved in (True, False):
            if interleaved:
                covariance = latent_covar
            else:
                order = torch.tensor([0, 2, 1, 3])
                covariance = latent_covar.index_select(-2, order).index_select(-1, order)
            latent = MultitaskMultivariateNormal(mean, covariance, interleaved=interleaved)
            for method, reference in references.items():
                if method == "forward" and not interleaved:
                    continue
                with self.subTest(interleaved=interleaved, method=method):
                    if method == "forward":
                        actual = likelihood.forward(mean).log_prob(target)
                    else:
                        actual = getattr(likelihood, method)(target, latent)
                    torch.testing.assert_close(actual, reference)
                    actual_grads = torch.autograd.grad(actual.sum(), parameters, retain_graph=True, allow_unused=True)
                    reference_grads = torch.autograd.grad(
                        reference.sum(), parameters, retain_graph=True, allow_unused=True
                    )
                    for parameter, actual_grad, reference_grad in zip(parameters, actual_grads, reference_grads):
                        if actual_grad is None:
                            actual_grad = torch.zeros_like(parameter)
                        if reference_grad is None:
                            reference_grad = torch.zeros_like(parameter)
                        torch.testing.assert_close(actual_grad, reference_grad)

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
        covar = covar.expand(*batch_shape, *covar.matrix_shape)
        return MultitaskMultivariateNormal(torch.randn(*batch_shape, 5, 4), covar, interleaved=False)


class TestMultitaskGaussianLikelihoodBatch(TestMultitaskGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2, batch_shape=torch.Size([3])).eval()

    def test_nonbatch(self):
        pass
