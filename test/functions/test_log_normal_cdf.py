#!/usr/bin/env python3

import math
import unittest

import torch
from torch import nn

import gpytorch


class TestLogNormalCDF(unittest.TestCase):
    def test_forward(self):
        inputs = torch.tensor([-6, -5, -3, -1, 0, 1, 3, 5], dtype=torch.float)
        output = gpytorch.log_normal_cdf(inputs)

        # Answers should be reasonable for small values
        self.assertLess(math.fabs(output[0] + 20.7368), 1e-4)
        self.assertLess(math.fabs(output[1] + 15), 0.1)
        self.assertLess(math.fabs(output[2] + 6.6), 0.01)
        self.assertLess(math.fabs(output[3] + 1.841), 0.001)

        # Should be very accurate for positive values
        self.assertLess(math.fabs(output[4] + 0.693147), 1e-4)
        self.assertLess(math.fabs(output[5] + 0.1727), 1e-4)
        self.assertLess(math.fabs(output[6] + 0.00135081), 1e-4)
        self.assertLess(math.fabs(output[7] + 2.86652e-7), 1e-4)

    def test_backward(self):
        inputs = nn.Parameter(torch.tensor([-6, -5, -3, -1, 0, 1, 3, 5], dtype=torch.float))
        output = gpytorch.log_normal_cdf(inputs)
        output.backward(torch.ones(8))

        gradient = inputs.grad
        expected_gradient = torch.tensor(
            [6.1585, 5.1865, 3.2831, 1.5251, 0.7979, 0.2876, 0.0044, 0.0000], dtype=torch.float
        )

        # Should be reasonable for small values
        for d in torch.abs(gradient[:3] - expected_gradient[:3]):
            self.assertLess(d, 5e-1)

        # Should be very accurate for larger ones
        for d in torch.abs(gradient[3:] - expected_gradient[3:]):
            self.assertLess(d, 5e-4)


if __name__ == "__main__":
    unittest.main()
