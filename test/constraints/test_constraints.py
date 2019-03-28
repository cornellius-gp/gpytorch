#!/usr/bin/env python3

import torch
import unittest
import gpytorch

from torch.nn.functional import softplus, sigmoid
from test._base_test_case import BaseTestCase


class TestInterval(unittest.TestCase, BaseTestCase):
    def test_transform_float_bounds(self):
        constraint = gpytorch.constraints.Interval(1., 5.)

        v = torch.tensor(-3.)

        value = constraint.transform(v)
        actual_value = (5. * sigmoid(v)) + 1.

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_bounds(self):
        constraint = gpytorch.constraints.Interval(1., 5.)

        v = torch.tensor(-3.)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(v, value)

    def test_transform_tensor_bounds(self):
        constraint = gpytorch.constraints.Interval(torch.tensor([1., 2.]), torch.tensor([3., 4.]))

        v = torch.tensor([-3., -2.])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = 3. * sigmoid(v[0]) + 1.
        actual_value[1] = 4. * sigmoid(v[1]) + 2.

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_tensor_bounds(self):
        constraint = gpytorch.constraints.Interval(torch.tensor([1., 2.]), torch.tensor([3., 4.]))

        v = torch.tensor([-3., -2.])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(v, value)


class TestGreaterThan(unittest.TestCase, BaseTestCase):
    def test_transform_float_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan(1.)

        v = torch.tensor(-3.)

        value = constraint.transform(v)
        actual_value = softplus(v) + 1.

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan([1., 2.])

        v = torch.tensor([-3., -2.])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = softplus(v[0]) + 1.
        actual_value[1] = softplus(v[1]) + 2.

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan(1.)

        v = torch.tensor(-3.)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan([1., 2.])

        v = torch.tensor([-3., -2.])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestLessThan(unittest.TestCase, BaseTestCase):
    def test_transform_float_less_than(self):
        constraint = gpytorch.constraints.LessThan(1.)

        v = torch.tensor(-3.)

        value = constraint.transform(v)
        actual_value = -softplus(-v) + 1.

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_less_than(self):
        constraint = gpytorch.constraints.LessThan([1., 2.])

        v = torch.tensor([-3., -2.])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = -softplus(-v[0]) + 1.
        actual_value[1] = -softplus(-v[1]) + 2.

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_less_than(self):
        constraint = gpytorch.constraints.LessThan(1.)

        v = torch.tensor(-3.)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_less_than(self):
        constraint = gpytorch.constraints.LessThan([1., 2.])

        v = torch.tensor([-3., -2.])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestPositive(unittest.TestCase, BaseTestCase):
    def test_transform_float_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor(-3.)

        value = constraint.transform(v)
        actual_value = softplus(v)

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor([-3., -2.])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = softplus(v[0])
        actual_value[1] = softplus(v[1])

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor(-3.)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_positive(self):
        constraint = gpytorch.constraints.Positive()

        v = torch.tensor([-3., -2.])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


if __name__ == "__main__":
    unittest.main()
