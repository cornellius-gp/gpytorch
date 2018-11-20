#!/usr/bin/env python3

import torch
import gpytorch
import unittest
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from test.models._model_test_case import VariationalModelTestCase


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        if train_x.dim() == 3:
            variational_distribution = CholeskyVariationalDistribution(train_x.size(-2), batch_size=train_x.size(0))
        else:
            variational_distribution = CholeskyVariationalDistribution(train_x.size(-2))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class TestVariationalGP(VariationalModelTestCase, unittest.TestCase):
    def create_model(self, train_data):
        model = GPClassificationModel(train_data)
        return model

    def create_test_data(self):
        return torch.randn(50, 1)

    def create_likelihood_and_labels(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        labels = torch.randn(50) + 2
        return likelihood, labels

    def create_batch_test_data(self):
        return torch.randn(3, 50, 1)

    def create_batch_likelihood_and_labels(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=3)
        labels = torch.randn(3, 50) + 2
        return likelihood, labels


if __name__ == "__main__":
    unittest.main()
