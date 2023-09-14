#!/usr/bin/env python3

import unittest

import torch

from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors.torch_priors import GammaPrior


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y):
        batch_shape = train_x.shape[:-2]
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=batch_shape,
            noise_constraint=GreaterThan(
                1e-4,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1],
                batch_shape=batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestExactMarginalLogLikelihood(unittest.TestCase):
    def test_batched_eval(self):
        train_x = torch.rand(10, 2)
        train_y = torch.randn(10)
        non_batch_model = ExactGPModel(train_x, train_y)
        mll = ExactMarginalLogLikelihood(non_batch_model.likelihood, non_batch_model)
        output = non_batch_model(train_x)
        non_batch_mll_eval = mll(output, train_y)

        train_x = train_x.expand(10, -1, -1)
        train_y = train_y.expand(10, -1)
        batch_model = ExactGPModel(train_x, train_y)
        mll = ExactMarginalLogLikelihood(batch_model.likelihood, batch_model)
        output = batch_model(train_x)
        batch_mll_eval = mll(output, train_y)

        self.assertEqual(non_batch_mll_eval.shape, torch.Size())
        self.assertEqual(batch_mll_eval.shape, torch.Size([10]))
        self.assertTrue(torch.allclose(non_batch_mll_eval.expand(10), batch_mll_eval))

    def test_mll_computation(self):
        train_x, train_y = (torch.rand(10, 2), torch.rand(10))
        model = ExactGPModel(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        output = model(train_x)
        marginal_log_likelihood = mll(output, train_y)

        marginal_likelihood = model.likelihood(output)
        noise_prior = next(model.likelihood.named_priors())[2]
        outputscale_prior = next(model.covar_module.named_priors())[2]
        lengthscale_prior = next(model.covar_module.base_kernel.named_priors())[2]

        log_probs = [
            marginal_likelihood.log_prob(train_y),
            noise_prior.log_prob(model.likelihood.noise),
            outputscale_prior.log_prob(model.covar_module.outputscale),
            lengthscale_prior.log_prob(model.covar_module.base_kernel.lengthscale).sum(),
        ]
        marginal_log_likelihood_by_hand = sum(log_probs) / train_y.shape[0]

        self.assertTrue(torch.allclose(marginal_log_likelihood, marginal_log_likelihood_by_hand))
