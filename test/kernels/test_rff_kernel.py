#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import torch

import gpytorch
from gpytorch.kernels import RFFKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RFFKernel(num_samples=50))

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class TestRFFKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return RFFKernel(num_samples=5, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return RFFKernel(num_dims=num_dims, num_samples=7, ard_num_dims=num_dims, **kwargs)

    def test_active_dims_list(self):
        kernel = self.create_kernel_no_ard(active_dims=[0, 2, 4, 6])
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        randn_weights = kernel.randn_weights
        kernel_basic = self.create_kernel_no_ard()
        kernel_basic._init_weights(randn_weights=randn_weights)
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)

    def test_active_dims_range(self):
        active_dims = list(range(3, 9))
        kernel = self.create_kernel_no_ard(active_dims=active_dims)
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        randn_weights = kernel.randn_weights
        kernel_basic = self.create_kernel_no_ard()
        kernel_basic._init_weights(randn_weights=randn_weights)
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)

    def test_kernel_getitem_single_batch(self):
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([2]))
        x = self.create_data_single_batch()

        res1 = kernel(x).evaluate()[0]  # Result of first kernel on first batch of data
        randn_weights = kernel.randn_weights

        new_kernel = kernel[0]
        new_kernel._init_weights(randn_weights=randn_weights[0])
        res2 = new_kernel(x[0]).evaluate()  # Should also be result of first kernel on first batch of data.

        self.assertLess(torch.norm(res1 - res2) / res1.norm(), 1e-4)

    def test_kernel_getitem_double_batch(self):
        # TODO: Fix randomization
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([3, 2]))
        x = self.create_data_double_batch()

        res1 = kernel(x).evaluate()[0, 1]  # Result of first kernel on first batch of data
        randn_weights = kernel.randn_weights

        new_kernel = kernel[0, 1]
        new_kernel._init_weights(randn_weights=randn_weights[0, 1])
        res2 = new_kernel(x[0, 1]).evaluate()  # Should also be result of first kernel on first batch of data.

        self.assertLess(torch.norm(res1 - res2) / res1.norm(), 1e-4)

    def test_kernel_output_fewer_features_than_data(self):
        # not fixing the seed can result in occasional bad params that cause a flaky test
        torch.manual_seed(1234)
        train_x = torch.randn(1000, 3)
        train_y = torch.randn(1000)
        test_x = torch.randn(500, 3)
        model = TestModel(train_x, train_y)

        # Make sure that the prior kernel is the correct type
        model.train()
        output = model(train_x).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.LowRankRootLazyTensor)

        # Make sure that the prior predictive kernel is the correct type
        model.train()
        output = model.likelihood(model(train_x)).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.LowRankRootAddedDiagLazyTensor)

        # Make sure we're calling the correct prediction strategy
        _wrapped_ps = MagicMock(wraps=gpytorch.models.exact_prediction_strategies.RFFPredictionStrategy)
        with patch("gpytorch.models.exact_prediction_strategies.RFFPredictionStrategy", new=_wrapped_ps) as ps_mock:
            model.eval()
            output = model.likelihood(model(test_x))
            _ = output.mean + output.variance  # Compute something to break through any lazy evaluations
            self.assertTrue(ps_mock.called)

    def test_kernel_output_more_features_than_data(self):
        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        test_x = torch.randn(500, 3)
        model = TestModel(train_x, train_y)

        # Make sure that the prior kernel is the correct type
        model.train()
        output = model(train_x).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.RootLazyTensor)
        self.assertNotIsInstance(output, gpytorch.lazy.LowRankRootLazyTensor)

        # Make sure that the prior predictive kernel is the correct type
        model.train()
        output = model.likelihood(model(train_x)).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.AddedDiagLazyTensor)
        self.assertNotIsInstance(output, gpytorch.lazy.LowRankRootAddedDiagLazyTensor)

        # Make sure we're calling the correct prediction strategy
        _wrapped_ps = MagicMock(wraps=gpytorch.models.exact_prediction_strategies.RFFPredictionStrategy)
        with patch("gpytorch.models.exact_prediction_strategies.RFFPredictionStrategy", new=_wrapped_ps) as ps_mock:
            model.eval()
            output = model.likelihood(model(test_x))
            _ = output.mean + output.variance  # Compute something to break through any lazy evaluations
            self.assertTrue(ps_mock.called)
