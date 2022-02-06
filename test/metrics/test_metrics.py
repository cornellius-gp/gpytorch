import unittest

import torch

import gpytorch
from gpytorch.metrics import (
    average_coverage_error,
    mean_standardized_log_loss,
    negative_log_predictive_density,
    quantile_coverage_error,
)
from gpytorch.models import ExactGP

N_PTS = 50


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestMetrics(unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = ExactGPModel(train_x, train_y, likelihood)
        return model

    def create_test_data(self):
        return torch.randn(N_PTS, 1)

    def create_likelihood_and_labels(self, train_data):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        labels = torch.sin(train_data) + torch.randn_like(train_data) * 0.5
        return likelihood, labels.ravel()

    def get_metric(self, model, likelihood, train_data, labels, metric):
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            pred_dist = likelihood(model(train_data))
            return metric(pred_dist, labels)

    def check_metric(self, metric):
        train_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels(train_data)
        model = self.create_model(train_data, labels, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        init_value = self.get_metric(model, likelihood, train_data, labels, metric)

        model.train()
        likelihood.train()
        for i in range(20):
            optimizer.zero_grad()
            output = model(train_data)
            loss = -mll(output, labels)
            loss.backward()
            optimizer.step()

        final_value = self.get_metric(model, likelihood, train_data, labels, metric)
        self.assertLess(final_value, init_value)

    def test_nlpd(self):
        self.check_metric(negative_log_predictive_density)

    def test_msll(self):
        self.check_metric(mean_standardized_log_loss)

    def test_qce(self):
        self.check_metric(quantile_coverage_error)

    def test_ace(self):
        self.check_metric(average_coverage_error)
