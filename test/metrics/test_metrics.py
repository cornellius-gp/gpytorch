import math
import unittest

import torch

import gpytorch
from gpytorch.metrics import (
    average_coverage_error,
    mean_absolute_error,
    mean_squared_error,
    mean_standardized_log_loss,
    negative_log_predictive_density,
    quantile_coverage_error,
)
from gpytorch.models import ExactGP

N_PTS = 10


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactMultiTaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=2)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class TestMetricsSingleTask(unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = ExactGPModel(train_x, train_y, likelihood)
        return model

    def create_test_data(self):
        return torch.linspace(0, 1, N_PTS)

    def create_likelihood_and_labels(self, train_data):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        labels = torch.sin(train_data * (2 * math.pi)) + torch.randn(train_data.size()) * 0.2
        return likelihood, labels

    def get_metric(self, model, likelihood, train_data, labels, metric, **kwargs):
        model.eval()
        likelihood.eval()
        pred_dist = likelihood(model(train_data))
        return metric(pred_dist, labels, **kwargs)

    def check_metric(self, metric, **kwargs):
        train_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels(train_data)
        model = self.create_model(train_data, labels, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        init_value = self.get_metric(model, likelihood, train_data, labels, metric, **kwargs)

        model.train()
        likelihood.train()
        for _ in range(5):
            optimizer.zero_grad()
            output = model(train_data)
            loss = -mll(output, labels)
            loss.backward()
            optimizer.step()

        final_value = self.get_metric(model, likelihood, train_data, labels, metric, **kwargs)
        return init_value, final_value

    def test_mae(self):
        init_value, final_value = self.check_metric(mean_absolute_error)
        self.assertTrue(torch.all(final_value < init_value))
        self.assertTrue(final_value.requires_grad)  # check that the metric is differentiable

    def test_mse(self):
        init_value, final_value = self.check_metric(mean_squared_error)
        self.assertTrue(torch.all(final_value < init_value))
        self.assertTrue(final_value.requires_grad)  # check that the metric is differentiable

    def test_nlpd(self):
        init_value, final_value = self.check_metric(negative_log_predictive_density)
        self.assertLess(final_value, init_value)  # works for a scaler, for vector, use torch.all
        self.assertTrue(final_value.requires_grad)  # check that the metric is differentiable

    def test_msll(self):
        init_value, final_value = self.check_metric(mean_standardized_log_loss)
        self.assertTrue(torch.all(final_value < init_value))
        self.assertTrue(final_value.requires_grad)  # check that the metric is differentiable

    def test_qce(self):
        init_value, final_value = self.check_metric(quantile_coverage_error, quantile=95)
        self.assertTrue(torch.all(final_value <= init_value))  # strictly less may be unnecessary here

    def test_ace(self):
        init_value, final_value = self.check_metric(average_coverage_error)
        self.assertTrue(torch.all(final_value < init_value))


class TestMetricsMultiTask(TestMetricsSingleTask):
    def create_model(self, train_x, train_y, likelihood):
        model = ExactMultiTaskGPModel(train_x, train_y, likelihood)
        return model

    def create_test_data(self):
        return torch.linspace(0, 1, N_PTS)

    def create_likelihood_and_labels(self, train_data):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        labels = torch.stack(
            [
                torch.sin(train_data * (2 * math.pi)) + torch.randn(train_data.size()) * 0.2,
                torch.cos(train_data * (2 * math.pi)) + torch.randn(train_data.size()) * 0.2,
            ],
            -1,
        )
        return likelihood, labels
