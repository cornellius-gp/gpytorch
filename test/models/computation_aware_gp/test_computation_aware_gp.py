#!/usr/bin/env python3

import unittest

import torch

import gpytorch

from gpytorch.models import ComputationAwareGP
from gpytorch.models.computation_aware_gp.linear_solvers import PLS, policies
from gpytorch.test.model_test_case import BaseModelTestCase

N_PTS = 100


class ComputationAwareGPModel(ComputationAwareGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(
            train_x,
            train_y,
            likelihood,
            linear_solver=PLS(
                policy=policies.GradientPolicy(num_non_zero=16),
                max_iter=16,
            ),
        )  # TODO: parametrize with different policies
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestComputationAwareGP(BaseModelTestCase, unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = ComputationAwareGPModel(train_x, train_y, likelihood)
        return model

    def create_test_data(self):
        return torch.randn(N_PTS, 1)

    def create_likelihood_and_labels(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        labels = torch.randn(N_PTS) + 2
        return likelihood, labels

    def create_batch_test_data(self, batch_shape=torch.Size([3])):
        return torch.randn(*batch_shape, N_PTS, 1)

    def create_batch_likelihood_and_labels(self, batch_shape=torch.Size([3])):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
        labels = torch.randn(*batch_shape, N_PTS) + 2
        return likelihood, labels


if __name__ == "__main__":
    unittest.main()
