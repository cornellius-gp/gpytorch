#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.likelihoods import (
    BernoulliLikelihood,
    NonGaussianMultitaskLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
)


class _LMCModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks, num_latents):
        inducing_points = torch.rand(num_latents, 6, 1)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestNonGaussianMultitaskLikelihood(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n, self.t = 10, 3
        self.train_x = torch.linspace(0, 1, self.n)
        self.model = _LMCModel(num_tasks=self.t, num_latents=2)
        self.function_dist = self.model(self.train_x)

    def _check_elbo(self, base_likelihood, train_y, task_log_prob):
        likelihood = NonGaussianMultitaskLikelihood(base_likelihood)
        # one value per data point (summed over the tasks), as MultitaskGaussianLikelihood returns
        elp = likelihood.expected_log_prob(train_y, self.function_dist)
        self.assertEqual(elp.shape, torch.Size([self.n]))
        expected = torch.stack(
            [
                task_log_prob(train_y[:, i], self.function_dist.mean[:, i], self.function_dist.variance[:, i])
                for i in range(self.t)
            ],
            dim=-1,
        ).sum(-1)
        self.assertTrue(torch.allclose(elp, expected, atol=1e-4))

        # the ELBO is a scalar that can be backpropagated, and sums over every task and data point
        mll = gpytorch.mlls.VariationalELBO(likelihood, self.model, num_data=self.n)
        loss = -mll(self.function_dist, train_y)
        self.assertEqual(loss.shape, torch.Size([]))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in self.model.parameters()))

    def test_bernoulli(self):
        train_y = torch.bernoulli(torch.full((self.n, self.t), 0.5))

        def task_log_prob(y, mean, var):
            # the one-dimensional likelihood, applied to each task as if it were a separate GP with N points
            dist = gpytorch.distributions.MultivariateNormal(mean, torch.diag_embed(var))
            return BernoulliLikelihood().expected_log_prob(y, dist)

        self._check_elbo(BernoulliLikelihood(), train_y, task_log_prob)

    def test_poisson(self):
        train_y = torch.poisson(torch.full((self.n, self.t), 2.0))

        def task_log_prob(y, mean, var):
            dist = gpytorch.distributions.MultivariateNormal(mean, torch.diag_embed(var))
            return PoissonLikelihood().expected_log_prob(y, dist)

        self._check_elbo(PoissonLikelihood(), train_y, task_log_prob)

    def test_student_t(self):
        train_y = torch.randn(self.n, self.t)

        def task_log_prob(y, mean, var):
            dist = gpytorch.distributions.MultivariateNormal(mean, torch.diag_embed(var))
            return base.expected_log_prob(y, dist)

        base = StudentTLikelihood()
        self._check_elbo(base, train_y, task_log_prob)

    def test_log_marginal_and_marginal(self):
        train_y = torch.bernoulli(torch.full((self.n, self.t), 0.5))
        likelihood = NonGaussianMultitaskLikelihood(BernoulliLikelihood())
        self.model.eval()
        function_dist = self.model(self.train_x)
        self.assertEqual(likelihood.log_marginal(train_y, function_dist).shape, torch.Size([self.n]))
        pred = likelihood(function_dist)
        self.assertEqual(pred.probs.shape, torch.Size([self.n, self.t]))

    def test_wrapper_exposes_base_likelihood_parameters(self):
        likelihood = NonGaussianMultitaskLikelihood(StudentTLikelihood())
        names = [name for name, _ in likelihood.named_parameters()]
        self.assertIn("base_likelihood.raw_deg_free", names)
        self.assertIn("base_likelihood.raw_noise", names)

    def test_has_analytic_marginal(self):
        self.assertTrue(NonGaussianMultitaskLikelihood(BernoulliLikelihood()).has_analytic_marginal)
        self.assertFalse(NonGaussianMultitaskLikelihood(PoissonLikelihood()).has_analytic_marginal)


if __name__ == "__main__":
    unittest.main()
