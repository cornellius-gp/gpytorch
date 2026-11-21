#!/usr/bin/env python3

import unittest

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import GammaRobustVariationalELBO, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.priors import NormalPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class _SVGPModel(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, batch_shape: torch.Size = torch.Size()) -> None:
        vd = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=batch_shape)
        vs = VariationalStrategy(self, inducing_points, vd, learn_inducing_locations=False)
        super().__init__(vs)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = RBFKernel(
            ard_num_dims=inducing_points.size(-1),
            batch_shape=batch_shape,
            lengthscale_prior=NormalPrior(0.0, 1.0),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestApproximateMLL(unittest.TestCase):
    def test_beta_zero_does_not_divide_by_zero(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(8, 2, dtype=torch.double)
        y = torch.randn(8, dtype=torch.double)
        model = _SVGPModel(x[:4]).double().train()
        likelihood = GaussianLikelihood().double().train()

        elbo = VariationalELBO(likelihood, model, num_data=8, beta=0.0, combine_terms=False)
        log_lik, kl_div, log_prior = elbo(model(x), y)
        self.assertEqual(kl_div.item(), 0.0)
        combined = VariationalELBO(likelihood, model, num_data=8, beta=0.0)(model(x), y)
        torch.testing.assert_close(combined, log_lik + log_prior)

    def test_batched_prior_terms_keep_batch_independence(self) -> None:
        torch.manual_seed(1)
        batch_shape = torch.Size([3])
        x = torch.randn(3, 6, 2, dtype=torch.double)
        y = torch.randn(3, 6, dtype=torch.double)
        model = _SVGPModel(x[:, :3, :], batch_shape=batch_shape).double().train()
        model.covar_module.lengthscale = torch.tensor([[[0.3, 0.7]], [[0.8, 1.2]], [[1.5, 0.4]]], dtype=torch.double)
        likelihood = GaussianLikelihood(batch_shape=batch_shape).double().train()

        mll = VariationalELBO(likelihood, model, num_data=10, combine_terms=False)
        _, _, log_prior = mll(model(x), y)
        expected_prior = NormalPrior(0.0, 1.0).log_prob(model.covar_module.lengthscale).sum(dim=(-2, -1)) / 10.0
        torch.testing.assert_close(log_prior, expected_prior)

    def test_gamma_robust_elbo_positional_and_keyword_init(self) -> None:
        torch.manual_seed(2)
        x = torch.randn(6, 2, dtype=torch.double)
        y = torch.randn(6, dtype=torch.double)
        model = _SVGPModel(x[:3]).double().train()
        likelihood = GaussianLikelihood().double().train()

        mll_pos = GammaRobustVariationalELBO(likelihood, model, 6, 0.5, 1.05)
        mll_kw = GammaRobustVariationalELBO(likelihood, model, num_data=6, beta=0.5, gamma=1.05)
        out = model(x)
        torch.testing.assert_close(mll_pos(out, y), mll_kw(out, y))


if __name__ == "__main__":
    unittest.main()
