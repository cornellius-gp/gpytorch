import math
import unittest

import torch

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.metrics import (  # average_coverage_error,
    mean_absolute_error,
    mean_squared_error,
    mean_standardized_log_loss,
    negative_log_predictive_density,
    quantile_coverage_error,
    standardized_mean_squared_error,
)
from gpytorch.models import ExactGP


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ExactMultiTaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=2, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestMetricsSingleTask(unittest.TestCase):
    def setUp(self, batch_shape=None):
        train_x, train_y = self.create_data_and_labels(batch_shape=batch_shape)
        untrained_model = self.create_model(train_x, train_y)
        self.trained_model = self.train_model(untrained_model, train_x, train_y)
        self.untrained_model = self.create_model(train_x, train_y)
        self.testing_pts = self.create_data_and_labels(batch_shape=batch_shape, is_training=False)
        self.train_x = train_x
        self.train_y = train_y

    def create_model(self, train_x, train_y):
        likelihood = GaussianLikelihood(batch_shape=train_x.shape[:-2])
        model = ExactGPModel(train_x, train_y, likelihood)
        # assume determinisim
        return model

    def create_data_and_labels(self, n=20, batch_shape=None, seed=0, is_training=True):
        input_range = [0.0, 1.0] if is_training else [1.0, 1.2]
        torch.random.manual_seed(seed)
        if batch_shape is None:
            batch_shape = torch.Size()
        inputs = torch.linspace(*input_range, n).view(-1, 1).contiguous()
        if batch_shape != torch.Size():
            inputs = inputs.unsqueeze(0)
            inputs = inputs.repeat(*batch_shape, 1, 1)
        labels = torch.sin(inputs * (2 * math.pi)) + torch.randn_like(inputs) * 0.1
        return inputs, labels.squeeze()

    def get_metric(self, model, likelihood, train_data, labels, metric, **kwargs):
        model.eval()
        likelihood.eval()
        pred_dist = likelihood(model(train_data))
        return metric(pred_dist, labels, **kwargs)

    def train_model(self, model, train_x, train_y):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model.train()
        model.likelihood.train()
        for _ in range(10):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).sum()
            loss.backward()
            optimizer.step()

        return model

    def check_metric(self, metric, **kwargs):
        init_value = self.get_metric(
            self.untrained_model,
            self.untrained_model.likelihood,
            *self.testing_pts,
            metric,
            **kwargs,
        )
        final_value = self.get_metric(
            self.trained_model,
            self.trained_model.likelihood,
            *self.testing_pts,
            metric,
            **kwargs,
        )
        return init_value, final_value

    def _test_metric(self, metric, should_check_differentiable=True, **kwargs):
        init_value, final_value = self.check_metric(metric, **kwargs)
        self.assertEqual(final_value.shape, self.train_y.shape[:-1])
        self.assertTrue(torch.all(final_value <= init_value))
        if should_check_differentiable:
            self.assertTrue(final_value.requires_grad)  # check that the metric is differentiable

    def test_mean_absolute_error(self):
        self._test_metric(mean_absolute_error)

    def test_mean_squared_error(self):
        self._test_metric(mean_squared_error)

    def test_negative_log_predictive_density(self):
        self._test_metric(negative_log_predictive_density)

    def test_mean_standardized_log_loss(self):
        self._test_metric(mean_standardized_log_loss)

    def test_standardized_mean_squared_error(self):
        self._test_metric(standardized_mean_squared_error)

    def test_quantile_coverage_error(self):
        self._test_metric(
            quantile_coverage_error,
            should_check_differentiable=False,
            quantile=95.0,
        )
        # check that negative or very positive quantile coverages raise errors
        with self.assertRaises(NotImplementedError):
            self.get_metric(
                self.untrained_model,
                self.untrained_model.likelihood,
                *self.testing_pts,
                quantile_coverage_error,
                quantile=-1.0,
            )
        with self.assertRaises(NotImplementedError):
            self.get_metric(
                self.untrained_model,
                self.untrained_model.likelihood,
                *self.testing_pts,
                quantile_coverage_error,
                quantile=1000.0,
            )


class TestMetricsBatchedSingleTask(TestMetricsSingleTask):
    def setUp(self):
        batch_shape = torch.Size((4,))
        super().setUp(batch_shape=batch_shape)


class TestMetricsMultiTask(TestMetricsSingleTask):
    def create_data_and_labels(self, n=20, batch_shape=None, seed=0, is_training=True):
        input_range = [0.0, 1.0] if is_training else [1.0, 1.2]
        torch.random.manual_seed(seed)
        if batch_shape is None:
            batch_shape = torch.Size()
        inputs = torch.linspace(*input_range, n).view(-1, 1).contiguous()
        if batch_shape != torch.Size():
            inputs = inputs.unsqueeze(0)
            inputs = inputs.repeat(*batch_shape, 1, 1)
        labels = torch.cat(
            [
                torch.sin(inputs * (2 * math.pi)) + torch.randn_like(inputs) * 0.1,
                torch.cos(inputs * (2 * math.pi)) + torch.randn_like(inputs) * 0.1,
            ],
            -1,
        )
        return inputs, labels.squeeze().contiguous()

    def create_model(self, train_x, train_y):
        likelihood = MultitaskGaussianLikelihood(num_tasks=2)
        model = ExactMultiTaskGPModel(train_x, train_y, likelihood)
        return model

    def _test_metric(self, metric, should_check_differentiable=True, **kwargs):
        init_value, final_value = self.check_metric(metric, **kwargs)
        # here the trailing dimension is the number of tasks
        self.assertEqual(
            final_value.shape,
            torch.Size(
                (
                    *self.train_y.shape[:-2],
                    self.train_y.shape[-1],
                )
            ),
        )
        self.assertTrue(torch.all(final_value.mean() <= init_value.mean()))
        if should_check_differentiable:
            self.assertTrue(final_value.requires_grad)  # check that the metric is differentiable

    def test_negative_log_predictive_density(self):
        init_value, final_value = self.check_metric(negative_log_predictive_density)
        self.assertEqual(final_value.shape, self.train_y.shape[:-2])
        self.assertTrue(torch.all(final_value <= init_value))
        self.assertTrue(final_value.requires_grad)  # check that the metric is differentiable


class TestMetricsBatchedMultiTask(TestMetricsMultiTask):
    def setUp(self):
        batch_shape = torch.Size((4,))
        super().setUp(batch_shape=batch_shape)


class TestMetricStandardizationBatch(unittest.TestCase):
    @staticmethod
    def _fixture(
        dtype: torch.dtype, batch_shape: tuple[int, ...], multitask: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        tasks = 3 if multitask else 1
        parameter_shape = (*batch_shape, tasks) if multitask else batch_shape
        scales = torch.arange(torch.Size(parameter_shape).numel(), dtype=dtype).reshape(parameter_shape) + 1
        offsets = scales * 10
        data_dim = -2 if multitask else -1
        test_points = torch.arange(4, dtype=dtype)
        train_points = torch.arange(5, dtype=dtype)
        if multitask:
            test_points = test_points.unsqueeze(-1)
            train_points = train_points.unsqueeze(-1)
        test_y = offsets.unsqueeze(data_dim) + scales.unsqueeze(data_dim) * test_points
        train_y = offsets.unsqueeze(data_dim) + scales.unsqueeze(data_dim) * train_points
        mean = (test_y + 1).requires_grad_()
        variance = torch.full_like(mean, 2, requires_grad=True)
        return mean, variance, test_y, train_y, scales, offsets, data_dim

    @staticmethod
    def _distribution(mean: torch.Tensor, variance: torch.Tensor, layout: bool | None) -> MultivariateNormal:
        if layout is None:
            return MultivariateNormal(mean, torch.diag_embed(variance))
        flat_variance = variance if layout else variance.transpose(-1, -2)
        covariance = torch.diag_embed(flat_variance.flatten(-2))
        return MultitaskMultivariateNormal(mean, covariance, interleaved=layout)

    def test_smse_preserves_batch_and_task_statistics(self) -> None:
        for dtype in (torch.float32, torch.float64):
            for batch_shape in ((), (2,), (2, 1)):
                for layout in (None, False, True):
                    with self.subTest(dtype=dtype, batch=batch_shape, layout=layout):
                        mean, variance, test_y, _, scales, _, data_dim = self._fixture(
                            dtype,
                            batch_shape,
                            layout is not None,
                        )
                        distribution = self._distribution(mean, variance, layout)
                        # Sample variance of [0, 1, 2, 3] is 5/3, independently of offsets.
                        expected = (mean - test_y).square().mean(data_dim) / (scales.square() * (5 / 3))
                        actual = standardized_mean_squared_error(distribution, test_y)
                        self.assertEqual(actual.shape, expected.shape)
                        torch.testing.assert_close(actual, expected)
                        weights = torch.linspace(1, 2, expected.numel(), dtype=dtype).reshape(expected.shape)
                        actual_grad = torch.autograd.grad((actual * weights).sum(), mean, retain_graph=True)[0]
                        expected_grad = torch.autograd.grad((expected * weights).sum(), mean)[0]
                        torch.testing.assert_close(actual_grad, expected_grad)

    def test_msll_training_baseline_preserves_batch_and_task_statistics(self) -> None:
        for dtype in (torch.float32, torch.float64):
            for batch_shape in ((), (2,), (2, 1)):
                for layout in (None, False, True):
                    with self.subTest(dtype=dtype, batch=batch_shape, layout=layout):
                        mean, variance, test_y, train_y, scales, offsets, data_dim = self._fixture(
                            dtype,
                            batch_shape,
                            layout is not None,
                        )
                        distribution = self._distribution(mean, variance, layout)
                        model_loss = -torch.distributions.Normal(mean, variance.sqrt()).log_prob(test_y).mean(data_dim)
                        # Training values [0,1,2,3,4] have mean2 and sample variance5/2.
                        trivial_model = torch.distributions.Normal(
                            (offsets + 2 * scales).unsqueeze(data_dim),
                            (scales * math.sqrt(5 / 2)).unsqueeze(data_dim),
                        )
                        expected = model_loss + trivial_model.log_prob(test_y).mean(data_dim)
                        actual = mean_standardized_log_loss(distribution, test_y, train_y)
                        self.assertEqual(actual.shape, expected.shape)
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(mean_standardized_log_loss(distribution, test_y), model_loss)
                        weights = torch.linspace(1, 2, expected.numel(), dtype=dtype).reshape(expected.shape)
                        actual_grads = torch.autograd.grad(
                            (actual * weights).sum(),
                            (mean, variance),
                            retain_graph=True,
                        )
                        expected_grads = torch.autograd.grad((expected * weights).sum(), (mean, variance))
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(actual_grad, expected_grad)
