#!/usr/bin/env python3

import torch
import gpytorch
import unittest
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, WhitenedVariationalStrategy
from test.models._model_test_case import VariationalModelTestCase


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x, use_inducing=False, whitened=False):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(-2), batch_shape=train_x.shape[:-2])
        inducing_points = torch.randn(50, train_x.size(-1)) if use_inducing else train_x
        strategy_cls = WhitenedVariationalStrategy if whitened else VariationalStrategy
        variational_strategy = strategy_cls(
            self, inducing_points, variational_distribution, learn_inducing_locations=use_inducing
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class TestVariationalGP(VariationalModelTestCase, unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = GPClassificationModel(train_x)
        return model

    def create_test_data(self):
        return torch.randn(50, 1)

    def create_likelihood_and_labels(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        labels = torch.randn(50) + 2
        return likelihood, labels

    def create_batch_test_data(self, batch_shape=torch.Size([3])):
        return torch.randn(*batch_shape, 50, 1)

    def create_batch_likelihood_and_labels(self, batch_shape=torch.Size([3])):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
        labels = torch.randn(*batch_shape, 50) + 2
        return likelihood, labels


class TestSVGPVariationalGP(TestVariationalGP):
    def create_model(self, train_x, train_y, likelihood):
        model = GPClassificationModel(train_x, use_inducing=True)
        return model

    def test_backward_train_nochol(self):
        with gpytorch.settings.max_cholesky_size(0):
            self.test_backward_train()

    def test_batch_backward_train_nochol(self):
        with gpytorch.settings.max_cholesky_size(0):
            self.test_batch_backward_train()

    def test_multi_batch_backward_train_nochol(self):
        with gpytorch.settings.max_cholesky_size(0):
            self.test_multi_batch_backward_train()


class TestSVGPWhitenedVariationalGP(TestSVGPVariationalGP):
    def create_model(self, train_x, train_y, likelihood):
        model = GPClassificationModel(train_x, use_inducing=True, whitened=True)
        return model


if __name__ == "__main__":
    unittest.main()
