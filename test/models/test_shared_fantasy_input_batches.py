#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class BatchedSharedInputGP(ExactGP):
    def __init__(self, train_x, train_y):
        batch_shape = train_x.shape[:-2]
        super().__init__(train_x, train_y, GaussianLikelihood(batch_shape=batch_shape))
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestSharedFantasyInputBatches(unittest.TestCase):
    def _test_shared_batches(self, device):
        configurations = (
            ((2,), (), True),
            ((2, 3), (), False),
            ((2, 3), (), True),
            ((2, 3), (3,), True),
        )
        for batch_shape, input_batch, extra_fantasy_axis in configurations:
            for num_points in (1, 2):
                with self.subTest(
                    batch=batch_shape, input_batch=input_batch, fantasy=extra_fantasy_axis, points=num_points
                ):
                    self._compare_dense(device, batch_shape, input_batch, extra_fantasy_axis, num_points)

    def _compare_dense(self, device, batch_shape, input_batch, extra_fantasy_axis, num_points, shared_targets=False):
        train_x = torch.linspace(0.05, 0.9, 5, device=device, dtype=torch.double).unsqueeze(-1)
        train_x = train_x.expand(*batch_shape, 5, 1).clone()
        train_y = (train_x[..., 0] * 4).sin()
        model = BatchedSharedInputGP(train_x, train_y).to(train_x).eval()
        model.covar_module.base_kernel.lengthscale = 0.31
        model.covar_module.outputscale = 0.8
        model.likelihood.noise = 0.04
        new_x = torch.linspace(0.24, 0.7, num_points, device=device, dtype=torch.double).reshape(num_points, 1)
        new_x = new_x.expand(*input_batch, num_points, 1).clone().requires_grad_()
        target_batch = () if shared_targets else ((4,) if extra_fantasy_axis else ()) + batch_shape
        new_y = torch.linspace(
            0.1, 0.9, torch.Size(target_batch).numel() * num_points, device=device, dtype=torch.double
        )
        new_y = new_y.reshape(*target_batch, num_points).requires_grad_()
        query = train_x.new_tensor([[0.19], [0.77]]).requires_grad_()
        with gpytorch.settings.detach_test_caches(False), gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ):
            model(query)  # Initialize the original training caches.
            fantasy = model.get_fantasy_model(new_x, new_y)
            actual = fantasy(query)
            expected_mean, expected_covar = self._dense_posterior(model, new_x, new_y, query)
            torch.testing.assert_close(actual.mean, expected_mean, rtol=2e-9, atol=2e-10)
            torch.testing.assert_close(actual.covariance_matrix, expected_covar, rtol=2e-9, atol=2e-10)
            expected_batch = torch.broadcast_shapes(target_batch, batch_shape)
            self.assertEqual(fantasy.train_inputs[0].shape, (*expected_batch, 5 + num_points, 1))
            variables = (new_x, new_y, query)
            gradient = torch.autograd.grad(actual.mean.sin().sum() + actual.covariance_matrix.square().sum(), variables)
            expected_gradient = torch.autograd.grad(
                expected_mean.sin().sum() + expected_covar.square().sum(), variables
            )
            for own, expected in zip(gradient, expected_gradient, strict=True):
                torch.testing.assert_close(own, expected, rtol=3e-8, atol=3e-9)

    @staticmethod
    def _dense_posterior(model, new_x, new_y, query):
        batch = torch.broadcast_shapes(model.train_targets.shape[:-1], new_x.shape[:-2], new_y.shape[:-1])
        train_x, train_y = model.train_inputs[0], model.train_targets
        inputs = torch.cat(
            (
                train_x.expand(*batch, *train_x.shape[-2:]),
                new_x.expand(*batch, *new_x.shape[-2:]),
                query.expand(*batch, *query.shape),
            ),
            -2,
        )
        prior = model.forward(inputs)
        mean, covariance = prior.mean, prior.covariance_matrix
        observations = torch.cat((train_y.expand(*batch, train_y.shape[-1]), new_y.expand(*batch, new_y.shape[-1])), -1)
        n = observations.shape[-1]
        train_covar = covariance[..., :n, :n] + torch.diag_embed(model.likelihood.noise.expand(*batch, n))
        cross = covariance[..., n:, :n]
        conditional_mean = mean[..., n:] + (
            cross @ torch.linalg.solve(train_covar, (observations - mean[..., :n]).unsqueeze(-1))
        ).squeeze(-1)
        conditional_covar = covariance[..., n:, n:] - cross @ torch.linalg.solve(train_covar, cross.transpose(-1, -2))
        return conditional_mean, conditional_covar

    def test_shared_inputs_cpu(self):
        self._test_shared_batches(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_shared_inputs_cuda(self):
        self._test_shared_batches(torch.device("cuda"))

    def _test_existing_shapes(self, device):
        for batch, fantasy, shared_targets in (((2,), False, False), ((2,), False, True), ((), True, False)):
            for num_points in (1, 2):
                with self.subTest(batch=batch, fantasy=fantasy, shared_targets=shared_targets, points=num_points):
                    self._compare_dense(device, batch, (), fantasy, num_points, shared_targets)

    def test_existing_shapes_cpu(self):
        self._test_existing_shapes(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_existing_shapes_cuda(self):
        self._test_existing_shapes(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
