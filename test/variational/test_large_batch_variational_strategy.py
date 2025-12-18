import unittest

import torch

import gpytorch
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution
from gpytorch.variational.large_batch_variational_strategy import LargeBatchVariationalStrategy
from gpytorch.variational.variational_strategy import VariationalStrategy


class _GPModel(ApproximateGP):
    def __init__(
        self,
        inducing_points,
        variational_strategy_class: type[VariationalStrategy] = VariationalStrategy,
        random_initialization: bool = False,
    ):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = variational_strategy_class(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.variational_strategy.variational_params_initialized.fill_(1)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CustomVariationalStrategyMixin:
    def test_train_mode(self):
        torch.set_default_dtype(torch.float32)

        inducing_points = torch.rand(2, 2)
        train_x = torch.rand(5, 2)

        torch.manual_seed(42)
        model1 = _GPModel(
            inducing_points=inducing_points.clone(),
            variational_strategy_class=self.variational_strategy_class,
        )
        model1.train()
        output1 = model1(train_x)

        loss1 = output1.mean.mean() + output1.covariance_matrix.diag().mean()
        loss1.backward()

        torch.manual_seed(42)
        model2 = _GPModel(
            inducing_points=inducing_points.clone(),
            variational_strategy_class=VariationalStrategy,
        )
        model2.train()
        output2 = model2(train_x)

        loss2 = output2.mean.mean() + output2.covariance_matrix.diag().mean()
        loss2.backward()

        self.assertAllClose(output1.mean, output2.mean)
        self.assertAllClose(output1.covariance_matrix.diag(), output2.covariance_matrix.diag())

        self.assertAllClose(
            model1.mean_module.constant.grad,
            model2.mean_module.constant.grad,
        )
        self.assertAllClose(
            model1.covar_module.raw_lengthscale.grad,
            model2.covar_module.raw_lengthscale.grad,
        )
        self.assertAllClose(
            model1.variational_strategy.inducing_points.grad,
            model2.variational_strategy.inducing_points.grad,
            atol=1e-5,
        )
        self.assertAllClose(
            model1.variational_strategy._variational_distribution.variational_mean.grad,
            model2.variational_strategy._variational_distribution.variational_mean.grad,
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertAllClose(
            model1.variational_strategy._variational_distribution.chol_variational_covar.grad,
            model2.variational_strategy._variational_distribution.chol_variational_covar.grad,
            atol=1e-4,
            rtol=1e-4,
        )


class TestLargeBatchVariationalStrategy(unittest.TestCase, BaseTestCase, CustomVariationalStrategyMixin):
    variational_strategy_class = LargeBatchVariationalStrategy
