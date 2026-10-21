#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class BroadcastPriorGP(ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        batch_shape = torch.Size([3])
        super().__init__(train_x, train_y, GaussianLikelihood(batch_shape=batch_shape))
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestFantasyBroadcastPrior(unittest.TestCase):
    def _test_broadcast_prior(self, device: torch.device) -> None:
        for num_points in (1, 2):
            for extra_fantasy_axis in (False, True):
                with self.subTest(num_points=num_points, extra_fantasy_axis=extra_fantasy_axis):
                    train_x = torch.linspace(0.1, 0.9, 6, device=device, dtype=torch.double).unsqueeze(-1)
                    train_y = (train_x[:, 0] * 3).sin()
                    model = BroadcastPriorGP(train_x, train_y).to(train_x).eval()
                    model.mean_module.constant = train_x.new_tensor([-0.3, 0.1, 0.5])
                    model.covar_module.base_kernel.lengthscale = train_x.new_tensor([0.25, 0.5, 0.9]).view(3, 1, 1)
                    model.covar_module.outputscale = train_x.new_tensor([0.7, 1.0, 1.3])
                    model.likelihood.noise = train_x.new_tensor([0.025, 0.035, 0.045]).unsqueeze(-1)
                    # Inputs share the three parameter batches, while targets
                    # differ across those batches. Both include another batch.
                    new_x = torch.linspace(0.2, 0.7, 2 * num_points, device=device, dtype=torch.double)
                    new_x = new_x.reshape(2, 1, num_points, 1).requires_grad_()
                    target_batch = (4, 2, 3) if extra_fantasy_axis else (2, 3)
                    new_y = (
                        torch.linspace(
                            0.2, 1.2, torch.Size(target_batch).numel() * num_points, device=device, dtype=torch.double
                        )
                        .reshape(*target_batch, num_points)
                        .requires_grad_()
                    )
                    query = train_x.new_tensor([[0.28], [0.62]]).requires_grad_()
                    with gpytorch.settings.detach_test_caches(False), gpytorch.settings.fast_computations(
                        covar_root_decomposition=False, log_prob=False, solves=False
                    ):
                        model(query)  # Initialize the prediction strategy.
                        fantasy = model.get_fantasy_model(new_x, new_y)
                        actual = fantasy(query.reshape(1, 1, 2, 1))
                        expected_mean, expected_covar = self._dense_posterior(model, new_x, new_y, query)
                        torch.testing.assert_close(actual.mean, expected_mean, atol=2e-10, rtol=2e-9)
                        torch.testing.assert_close(actual.covariance_matrix, expected_covar, atol=2e-10, rtol=2e-9)
                        targets = (new_x, new_y, query)
                        actual_gradient = torch.autograd.grad(
                            actual.mean.square().sum() + 0.2 * actual.covariance_matrix.square().sum(), targets
                        )
                        expected_gradient = torch.autograd.grad(
                            expected_mean.square().sum() + 0.2 * expected_covar.square().sum(), targets
                        )
                        for actual_grad, expected_grad in zip(actual_gradient, expected_gradient, strict=True):
                            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-9, rtol=2e-8)

    @staticmethod
    def _dense_posterior(
        model: BroadcastPriorGP, new_x: torch.Tensor, new_y: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = torch.broadcast_shapes(new_x.shape[:-2], new_y.shape[:-1], model.covar_module.batch_shape)
        train_x, train_y = model.train_inputs[0], model.train_targets
        full_x = torch.cat(
            (
                train_x.expand(*batch, *train_x.shape),
                new_x.expand(*batch, *new_x.shape[-2:]),
                query.expand(*batch, *query.shape),
            ),
            dim=-2,
        )
        prior = model.forward(full_x)
        prior_mean, prior_covar = prior.mean, prior.covariance_matrix
        observations = torch.cat((train_y.expand(*batch, len(train_y)), new_y.expand(*batch, new_y.shape[-1])), -1)
        count = observations.shape[-1]
        noise = model.likelihood.noise.expand(*batch, count)
        train_covar = prior_covar[..., :count, :count] + torch.diag_embed(noise)
        cross = prior_covar[..., count:, :count]
        mean = prior_mean[..., count:] + torch.matmul(
            cross, torch.linalg.solve(train_covar, (observations - prior_mean[..., :count]).unsqueeze(-1))
        ).squeeze(-1)
        covariance = prior_covar[..., count:, count:] - cross @ torch.linalg.solve(train_covar, cross.transpose(-1, -2))
        return mean, covariance

    def test_broadcast_prior_cpu(self) -> None:
        self._test_broadcast_prior(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_broadcast_prior_cuda(self) -> None:
        self._test_broadcast_prior(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
