#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch import settings
from gpytorch.kernels import GridInterpolationKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import InterpolatedPredictionStrategy
from gpytorch.test.model_test_case import BaseModelTestCase

N_PTS = 50


class GridInterpolationKernelMock(GridInterpolationKernel):
    def __init__(self, should_use_wiski=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_use_wiski = should_use_wiski

    def prediction_strategy(self, *args, **kwargs):
        return InterpolatedPredictionStrategy(uses_wiski=self.should_use_wiski, *args, **kwargs)


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
    def __init__(self, train_x, train_y, likelihood, should_use_wiski=False):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = GridInterpolationKernelMock(
            base_kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            grid_size=128,
            num_dims=1,
            should_use_wiski=should_use_wiski,
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
        covar_c = gpytorch.kernels.LinearKernel()  # this one is important because its covariance matrix can be lazy
        self.covar_module = covar_a + covar_b + covar_c

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestExactGP(BaseModelTestCase, unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = ExactGPModel(train_x, train_y, likelihood)
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

    def test_lanczos_fantasy_model(self):
        lanczos_thresh = 10
        n = lanczos_thresh + 1
        n_dims = 2
        with settings.max_cholesky_size(lanczos_thresh):
            x = torch.ones((n, n_dims))
            y = torch.randn(n)
            likelihood = GaussianLikelihood()
            model = ExactGPModel(x, y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            mll.train()
            mll.eval()

            # get a posterior to fill in caches
            model(torch.randn((1, n_dims)))

            new_n = 2
            new_x = torch.randn((new_n, n_dims))
            new_y = torch.randn(new_n)
            # just check that this can run without error
            model.get_fantasy_model(new_x, new_y)


class TestInterpolatedExactGP(TestExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = InterpolatedExactGPModel(train_x, train_y, likelihood)
        return model


class TestWiskiExactGP(TestInterpolatedExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = InterpolatedExactGPModel(train_x, train_y, likelihood, should_use_wiski=True)
        return model

    def test_fantasy_model(self):
        x = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(x, labels, likelihood)
        test_x = self.create_test_data()
        _, test_labels = self.create_likelihood_and_labels()
        with torch.no_grad():
            model.eval()
            model(test_x)
        new_model = model.get_fantasy_model(test_x, test_labels)

        self.assertEqual(type(new_model), type(model))
        self.assertTrue(new_model.prediction_strategy.uses_wiski)

    def test_nonbatch_to_batch_fantasy_model(self, batch_shape=torch.Size([3])):
        x = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(x, labels, likelihood)
        test_x = self.create_batch_test_data(batch_shape=batch_shape)
        _, test_labels = self.create_batch_likelihood_and_labels(batch_shape=batch_shape)
        with torch.no_grad():
            model.eval()
            model(test_x)
        new_model = model.get_fantasy_model(test_x, test_labels)

        self.assertEqual(type(new_model), type(model))
        self.assertTrue(new_model.prediction_strategy.uses_wiski)

    def test_nonbatch_to_multibatch_fantasy_model(self):
        self.test_nonbatch_to_batch_fantasy_model(batch_shape=torch.Size([2, 3]))


class TestSumExactGP(TestExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = SumExactGPModel(train_x, train_y, likelihood)
        return model

    def test_cache_across_lazy_threshold(self):
        x = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(x, labels, likelihood)
        model.eval()
        model(x)  # populate caches

        with settings.max_eager_kernel_size(2 * N_PTS - 1), settings.fast_pred_var(True):
            # now we'll cross the threshold and use lazy tensors
            new_x = self.create_test_data()
            _, new_y = self.create_likelihood_and_labels()
            model = model.get_fantasy_model(new_x, new_y)
            predicted = model(self.create_test_data())

            # the main purpose of the test was to ensure there was no error, but we can verify shapes too
            self.assertEqual(predicted.mean.shape, torch.Size([N_PTS]))
            self.assertEqual(predicted.variance.shape, torch.Size([N_PTS]))


if __name__ == "__main__":
    unittest.main()
