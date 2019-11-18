#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.test.model_test_case import BaseModelTestCase


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class InterpolatedExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), grid_size=128, num_dims=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SumExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        covar_a = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        covar_b = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        self.covar_module = covar_a + covar_b

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestExactGP(BaseModelTestCase, unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = ExactGPModel(train_x, train_y, likelihood)
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

    def test_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_forward_eval()

    def test_batch_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_batch_forward_eval()

    def test_multi_batch_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_multi_batch_forward_eval()

    def test_batch_forward_then_nonbatch_forward_eval(self):
        batch_data = self.create_batch_test_data()
        likelihood, labels = self.create_batch_likelihood_and_labels()
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)

        # Smoke test derivatives working
        output.mean.sum().backward()

        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Create non-batch data
        data = self.create_test_data()
        output = model(data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == data.size(-2))

        # Smoke test derivatives working
        output.mean.sum().backward()

    def test_batch_forward_then_different_batch_forward_eval(self):
        non_batch_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(non_batch_data, labels, likelihood)
        model.eval()

        # Batch size 3
        batch_data = self.create_batch_test_data()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now Batch size 2
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2]))
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 3 again
        batch_data = self.create_batch_test_data()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 1
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([1]))
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

    def test_prior_mode(self):
        train_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        prior_model = self.create_model(None, None, likelihood)
        model = self.create_model(train_data, labels, likelihood)
        prior_model.eval()
        model.eval()

        test_data = self.create_test_data()
        prior_out = prior_model(test_data)
        with gpytorch.settings.prior_mode(True):
            prior_out_cm = model(test_data)
        self.assertTrue(torch.allclose(prior_out.mean, prior_out_cm.mean))
        self.assertTrue(torch.allclose(prior_out.covariance_matrix, prior_out_cm.covariance_matrix))


class TestInterpolatedExactGP(TestExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = InterpolatedExactGPModel(train_x, train_y, likelihood)
        return model


class TestSumExactGP(TestExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = SumExactGPModel(train_x, train_y, likelihood)
        return model


if __name__ == "__main__":
    unittest.main()
