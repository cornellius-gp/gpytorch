#!/usr/bin/env python3

import unittest

import torch
from torch import sigmoid
from torch.nn.functional import softplus

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase


# Basic exact GP model for testing parameter + constraint name resolution
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestInterval(unittest.TestCase, BaseTestCase):
    def test_transform_float_bounds(self):
        constraint = gpytorch.constraints.Interval(1.0, 5.0)

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = ((5.0 - 1.0) * sigmoid(v)) + 1.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_bounds(self):
        constraint = gpytorch.constraints.Interval(1.0, 5.0)

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(v, value)

    def test_transform_tensor_bounds(self):
        constraint = gpytorch.constraints.Interval(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = (3.0 - 1.0) * sigmoid(v[0]) + 1.0
        actual_value[1] = (4.0 - 2.0) * sigmoid(v[1]) + 2.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_tensor_bounds(self):
        constraint = gpytorch.constraints.Interval(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(v, value)

    def test_initial_value(self):
        constraint = gpytorch.constraints.Interval(1.0, 5.0, transform=None, initial_value=3.0)
        lkhd = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=constraint)
        self.assertEqual(lkhd.noise.item(), 3.0)


class TestGreaterThan(unittest.TestCase, BaseTestCase):
    def test_transform_float_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = softplus(v) + 1.0

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = softplus(v[0]) + 1.0
        actual_value[1] = softplus(v[1]) + 2.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestLessThan(unittest.TestCase, BaseTestCase):
    def test_transform_float_less_than(self):
        constraint = gpytorch.constraints.LessThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = -softplus(-v) + 1.0

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_less_than(self):
        constraint = gpytorch.constraints.LessThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = -softplus(-v[0]) + 1.0
        actual_value[1] = -softplus(-v[1]) + 2.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_less_than(self):
        constraint = gpytorch.constraints.LessThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_less_than(self):
        constraint = gpytorch.constraints.LessThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestPositive(unittest.TestCase, BaseTestCase):
    def test_transform_float_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = softplus(v)

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = softplus(v[0])
        actual_value[1] = softplus(v[1])

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestConstraintNaming(unittest.TestCase, BaseTestCase):
    def test_constraint_by_name(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(None, None, likelihood)

        constraint = model.constraint_for_parameter_name("likelihood.noise_covar.raw_noise")
        self.assertIsInstance(constraint, gpytorch.constraints.GreaterThan)

        constraint = model.constraint_for_parameter_name("covar_module.base_kernel.raw_lengthscale")
        self.assertIsInstance(constraint, gpytorch.constraints.Positive)

        constraint = model.constraint_for_parameter_name("mean_module.constant")
        self.assertIsNone(constraint)

    def test_named_parameters_and_constraints(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(None, None, likelihood)

        for name, _param, constraint in model.named_parameters_and_constraints():
            if name == "likelihood.noise_covar.raw_noise":
                self.assertIsInstance(constraint, gpytorch.constraints.GreaterThan)
            elif name == "mean_module.constant":
                self.assertIsNone(constraint)
            elif name == "covar_module.raw_outputscale":
                self.assertIsInstance(constraint, gpytorch.constraints.Positive)
            elif name == "covar_module.base_kernel.raw_lengthscale":
                self.assertIsInstance(constraint, gpytorch.constraints.Positive)


if __name__ == "__main__":
    unittest.main()
