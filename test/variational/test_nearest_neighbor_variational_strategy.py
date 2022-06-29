#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import torch

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase


from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy

model_batch_shape = torch.Size([])
data_batch_shape = torch.Size([])
k = 3
train_n = 10
D = 2
train_x = torch.randn(data_batch_shape + torch.Size([train_n, D]))
train_y = torch.randn(data_batch_shape + torch.Size([train_n, ]))

class NNGPModel(gpytorch.models.GP):
    def __init__(self, inducing_points, likelihood, k=32, kernel_type='mat52-ard',
                 batch_shape=torch.Size([]),
                 os_init=None, ell_init=None, noise_init=None,
                 learn_kernel=True, learn_noise=True,
                 training_batch_size=256):
        super().__init__()

        inducing_batch_shape = inducing_points.shape[:-2]
        m, d = inducing_points.shape[-2:]
        self.m = m
        self.k = k

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m, batch_shape=batch_shape)

        if torch.cuda.is_available():
            inducing_points = inducing_points.cuda()

        variational_strategy = NNVariationalStrategy(self, inducing_points, variational_distribution, k=k,
                                                     training_batch_size=training_batch_size)
        self.variational_strategy = variational_strategy
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
        self.likelihood = likelihood

        self.has_likelihood_noise = hasattr(self.likelihood, 'noise')
        if self.has_likelihood_noise:
            if noise_init is not None:
                self.likelihood.initialize(noise=noise_init)
            if not learn_noise:
                self.likelihood.noise_covar.raw_noise.requires_grad = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=False, **kwargs)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = NNGPModel(inducing_points=train_x.clone(), likelihood=likelihood, k=3, batch_shape=model_batch_shape,
                  training_batch_size=5)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_n)
model.train()
firstk_kl = model.variational_strategy.firstk_kl_helper()
kl_indices = model.variational_strategy._get_training_indices()# first kl
kl_indices = model.variational_strategy._get_training_indices()
print(kl_indices)
stochastic_kl = model.variational_strategy._stochastic_kl_helper_no_batch(kl_indices)

"""
model.train()
output = model(x=None)
current_training_indices = model.variational_strategy.current_training_indices
print(current_training_indices.shape)
ybatch = train_y[current_training_indices]
loss =  -mll(output, ybatch).mean()

output = model(x=None)
current_training_indices = model.variational_strategy.current_training_indices
print(current_training_indices.shape)
ybatch = train_y[current_training_indices]
loss =  -mll(output, ybatch).mean()
"""


"""
class TestVNNGP(BaseTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.NNVariationalStrategy

    @property
    def likelihood_cls(self):
        return gpytorch.likelihoods.GaussianLikelihood

    @property
    def cuda(self):
        return False

    def _make_model_and_likelihood(
        self,
        inducing_points,
        number_of_nearest_neighbors,
        strategy_cls=gpytorch.variational.NNVariationalStrategy,
        distribution_cls=gpytorch.variational.MeanFieldVariationalDistribution
    ):

        class VNNGPRegressionModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points, k=256, training_batch_size=256):

                m, d = inducing_points.shape
                self.m = m
                self.k = k

                variational_distribution = distribution_cls(m)

                if torch.cuda.is_available():
                    inducing_points = inducing_points.cuda()

                variational_strategy = strategy_cls(self, inducing_points, variational_distribution, k=k,
                                                             training_batch_size=training_batch_size)
                super(VNNGPRegressionModel, self).__init__(variational_strategy)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            def __call__(self, x, prior=False, **kwargs):
                if x is not None:
                    if x.dim() == 1:
                        x = x.unsqueeze(-1)
                return self.variational_strategy(x=x, prior=False, **kwargs)

        return VNNGPRegressionModel(inducing_points, k=number_of_nearest_neighbors), self.likelihood_cls()

    def test_eval_iteration(self):
        # Mocks
        _wrapped_cholesky = MagicMock(wraps=torch.linalg.cholesky_ex)
        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        _wrapped_ciq = MagicMock(wraps=gpytorch.utils.contour_integral_quad)
        _cholesky_mock = patch("torch.linalg.cholesky_ex", new=_wrapped_cholesky)
        _cg_mock = patch("gpytorch.utils.linear_cg", new=_wrapped_cg)
        _ciq_mock = patch("gpytorch.utils.contour_integral_quad", new=_wrapped_ciq)

        _wrapper_vnngp = MagicMock(wraps=gpytorch.utils.NNUtil)
        _vnngp_mock = patch("gpytorch.utils.NNUtil", new=_wrapper_vnngp)

        # Make model and likelihood
        inducing_points = torch.randn(100, 2)
        k = 5
        model, likelihood = self._make_model_and_likelihood(
            inducing_points=inducing_points,
            number_of_nearest_neighbors=k,
            strategy_cls=self.strategy_cls,
            distribution_cls=self.distribution_cls
        )

        # Do one forward pass
        self._training_iter(model, likelihood, data_batch_shape, mll_cls=self.mll_cls, cuda=self.cuda)

        # Now do evaluation
        with _cholesky_mock as cholesky_mock, _cg_mock as cg_mock, _ciq_mock as ciq_mock:
            # Iter 1
            _ = self._eval_iter(model, eval_data_batch_shape, cuda=self.cuda)
            output = self._eval_iter(model, eval_data_batch_shape, cuda=self.cuda)
            self.assertEqual(output.batch_shape, expected_batch_shape)
            self.assertEqual(output.event_shape, self.event_shape)
            return cg_mock, cholesky_mock, ciq_mock

    def test_eval_smaller_pred_batch(self):
        return self.test_eval_iteration(
            model_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3, 1]) + self.batch_shape),
            data_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            eval_data_batch_shape=(torch.Size([4]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
        )

    def test_eval_larger_pred_batch(self):
        return self.test_eval_iteration(
            model_batch_shape=(torch.Size([4]) + self.batch_shape),
            inducing_batch_shape=(self.batch_shape),
            data_batch_shape=(torch.Size([4]) + self.batch_shape),
            eval_data_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
        )

    def test_training_iteration(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        expected_batch_shape=None,
        constant_mean=True,
    ):
        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        data_batch_shape = data_batch_shape if data_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape

        # Mocks
        _wrapped_cholesky = MagicMock(wraps=torch.linalg.cholesky_ex)
        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        _wrapped_ciq = MagicMock(wraps=gpytorch.utils.contour_integral_quad)
        _cholesky_mock = patch("torch.linalg.cholesky_ex", new=_wrapped_cholesky)
        _cg_mock = patch("gpytorch.utils.linear_cg", new=_wrapped_cg)
        _ciq_mock = patch("gpytorch.utils.contour_integral_quad", new=_wrapped_ciq)

        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
            distribution_cls=self.distribution_cls,
            strategy_cls=self.strategy_cls,
            constant_mean=constant_mean,
        )

        # Do forward pass
        with _cholesky_mock as cholesky_mock, _cg_mock as cg_mock, _ciq_mock as ciq_mock:
            # Iter 1
            self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 0)
            self._training_iter(
                model,
                likelihood,
                data_batch_shape,
                mll_cls=self.mll_cls,
                cuda=self.cuda,
            )
            self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 1)
            # Iter 2
            output, loss = self._training_iter(
                model,
                likelihood,
                data_batch_shape,
                mll_cls=self.mll_cls,
                cuda=self.cuda,
            )
            self.assertEqual(output.batch_shape, expected_batch_shape)
            self.assertEqual(output.event_shape, self.event_shape)
            self.assertEqual(loss.shape, expected_batch_shape)
            return cg_mock, cholesky_mock, ciq_mock

    def test_training_iteration_batch_inducing(self):
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            data_batch_shape=self.batch_shape,
            inducing_batch_shape=(torch.Size([3]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_iteration_batch_data(self):
        return self.test_training_iteration(
            model_batch_shape=self.batch_shape,
            inducing_batch_shape=self.batch_shape,
            data_batch_shape=(torch.Size([3]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_iteration_batch_model(self):
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            inducing_batch_shape=self.batch_shape,
            data_batch_shape=self.batch_shape,
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_all_batch_zero_mean(self):
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3, 1]) + self.batch_shape),
            data_batch_shape=(torch.Size([4]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            constant_mean=False,
        )


if __name__ == "__main__":
    unittest.main()

"""