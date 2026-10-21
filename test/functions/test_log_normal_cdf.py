#!/usr/bin/env python3

import math
import unittest

import torch
from scipy.special import log_ndtr
from torch import nn

import gpytorch


class TestLogNormalCDF(unittest.TestCase):
    @property
    def devices(self) -> list[str]:
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

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

    def test_reference_values_across_branches(self) -> None:
        values = [-40.0, -20.0, -8.0, -2.0, -1.001, -0.999, -0.201, -0.199, 0.0, 0.201, 3.0, 10.0]
        for device in self.devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    inputs = torch.tensor(values, device=device, dtype=dtype).reshape(3, 4).mT
                    self.assertFalse(inputs.is_contiguous())
                    expected = torch.as_tensor(log_ndtr(values), device=device, dtype=dtype).reshape(3, 4).mT
                    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
                    torch.testing.assert_close(
                        gpytorch.log_normal_cdf(inputs), expected, atol=tolerance, rtol=tolerance
                    )

    def test_negative_tail_higher_derivatives(self) -> None:
        values = [-3.0, -2.0, -1.5, -0.2, 0.0, 0.4, 2.5]
        reference = torch.tensor(values, dtype=torch.double)
        log_cdf = torch.from_numpy(log_ndtr(values))
        ratio = (-0.5 * reference.square() - log_cdf - 0.5 * math.log(2 * math.pi)).exp()
        second = -ratio * (reference + ratio)
        third = -ratio - (reference + 2 * ratio) * second
        for device in self.devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    inputs = torch.tensor(values, device=device, dtype=dtype, requires_grad=True)
                    output = gpytorch.log_normal_cdf(inputs)
                    first = torch.autograd.grad(output.sum(), inputs, create_graph=True)[0]
                    actual_second = torch.autograd.grad(first.sum(), inputs, create_graph=True)[0]
                    tolerance = 3e-5 if dtype == torch.float32 else 2e-11
                    torch.testing.assert_close(actual_second, second.to(inputs), atol=tolerance, rtol=tolerance)
                    if dtype == torch.double:
                        actual_third = torch.autograd.grad(actual_second.sum(), inputs)[0]
                        torch.testing.assert_close(actual_third, third.to(inputs), atol=2e-10, rtol=2e-10)

    def test_gradcheck_and_gradgradcheck(self) -> None:
        for device in self.devices:
            with self.subTest(device=device):
                inputs = torch.tensor(
                    [[-8.0, -2.0, -1.1], [-0.1, 0.3, 2.0]],
                    device=device,
                    dtype=torch.double,
                ).mT.requires_grad_()
                self.assertTrue(torch.autograd.gradcheck(gpytorch.log_normal_cdf, (inputs,)))
                self.assertTrue(torch.autograd.gradgradcheck(gpytorch.log_normal_cdf, (inputs,)))
                self.assertTrue(
                    torch.autograd.gradgradcheck(lambda x: gpytorch.log_normal_cdf(x).square().sum(), (inputs,))
                )

    def test_probit_likelihood_gradgradcheck(self) -> None:
        for device in self.devices:
            with self.subTest(device=device):
                likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device=device, dtype=torch.double)
                mean = torch.tensor([[-2.0, -1.3], [0.2, 0.5]], device=device, dtype=torch.double, requires_grad=True)
                raw_variance = torch.full_like(mean, -0.7, requires_grad=True)
                target = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device, dtype=torch.double)

                def expected_log_prob(location: torch.Tensor, raw_scale: torch.Tensor) -> torch.Tensor:
                    variance = raw_scale.exp()
                    distribution = gpytorch.distributions.MultivariateNormal(location, torch.diag_embed(variance))
                    return likelihood.expected_log_prob(target, distribution).sum()

                self.assertTrue(torch.autograd.gradcheck(expected_log_prob, (mean, raw_variance)))
                self.assertTrue(torch.autograd.gradgradcheck(expected_log_prob, (mean, raw_variance)))


if __name__ == "__main__":
    unittest.main()
