#!/usr/bin/env python3

import math
import unittest
from unittest.mock import Mock

import torch

import gpytorch


def dist_func(x1, x2):
    dist_module = gpytorch.kernels.kernel.Distance()
    return dist_module._dist(x1, x2)


class TestMaternCovariance(unittest.TestCase):
    def test_lengthscale_gradgradcheck_and_third_derivative(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for nu in (0.5, 1.5, 2.5):
                for batch_shape in ((), (2,), (2, 1)):
                    for same_inputs in (False, True):
                        with self.subTest(device=device, nu=nu, batch_shape=batch_shape, same_inputs=same_inputs):
                            x1 = torch.tensor(
                                [[0.03, 0.2], [0.29, 0.6], [0.58, 0.1], [0.87, 0.8]],
                                dtype=torch.double,
                                device=device,
                            )
                            x2 = x1 if same_inputs else x1[:3].flip(0) + 0.07
                            lengthscale = torch.full(
                                (*batch_shape, 1, 1),
                                0.47,
                                dtype=torch.double,
                                device=device,
                                requires_grad=True,
                            )

                            def kernel(value: torch.Tensor) -> torch.Tensor:
                                return gpytorch.functions.MaternCovariance.apply(x1, x2, value, nu, dist_func)

                            self.assertTrue(torch.autograd.gradcheck(kernel, (lengthscale,)))
                            self.assertTrue(torch.autograd.gradgradcheck(kernel, (lengthscale,)))
                            # A constant Euclidean distance has no input-gradient
                            # graph; this oracle does not use the custom function.
                            distance = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(-1).sqrt()
                            scaled = math.sqrt(2 * nu) * distance / lengthscale
                            polynomial = 1 if nu == 0.5 else 1 + scaled
                            if nu == 2.5:
                                polynomial = polynomial + scaled.square() / 3
                            actual, expected = kernel(lengthscale).sum(), (polynomial * (-scaled).exp()).sum()
                            for _ in range(3):
                                actual = torch.autograd.grad(actual.sum(), lengthscale, create_graph=True)[0]
                                expected = torch.autograd.grad(expected.sum(), lengthscale, create_graph=True)[0]
                                torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8)

    def test_first_order_uses_cached_derivative(self) -> None:
        x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], dtype=torch.double)
        for nu in (0.5, 1.5, 2.5):
            lengthscale = torch.tensor([[0.31]], dtype=torch.double, requires_grad=True)
            distance = Mock(side_effect=dist_func)
            value = gpytorch.functions.MaternCovariance.apply(x, x, lengthscale, nu, distance)
            torch.autograd.grad(value.sum(), lengthscale)
            self.assertEqual(distance.call_count, 1)

    def test_large_coordinate_normalization_and_infinite_lengthscale(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for nu in (0.5, 1.5, 2.5):
                with self.subTest(device=device, nu=nu):
                    scale = 1e160
                    unit_x = torch.tensor([[0.1], [0.3], [0.6], [0.9]], dtype=torch.double, device=device)
                    x = unit_x * scale
                    lengthscale = (
                        torch.tensor(
                            [scale, float("inf")],
                            dtype=torch.double,
                            device=device,
                        )
                        .reshape(2, 1, 1)
                        .requires_grad_()
                    )
                    actual = gpytorch.functions.MaternCovariance.apply(x, x, lengthscale, nu, dist_func)
                    # Distances in the original units would overflow while
                    # these normalized inputs and the covariance stay finite.
                    distance = (unit_x.unsqueeze(-2) - unit_x.unsqueeze(-3)).abs().squeeze(-1)
                    scaled = math.sqrt(2 * nu) * distance / (lengthscale / scale)
                    polynomial = 1 if nu == 0.5 else 1 + scaled
                    if nu == 2.5:
                        polynomial = polynomial + scaled.square() / 3
                    expected = polynomial * (-scaled).exp()
                    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
                    actual_grad = torch.autograd.grad(actual.sum(), lengthscale, create_graph=True)[0]
                    expected_grad = torch.autograd.grad(expected.sum(), lengthscale)[0]
                    torch.testing.assert_close(actual_grad * scale, expected_grad * scale, atol=1e-12, rtol=1e-11)
                    self.assertEqual(actual_grad[1].item(), 0)
                    second = torch.autograd.grad(actual_grad.sum(), lengthscale)[0]
                    self.assertTrue(second.isfinite().all())

    def test_1_2_forward(self):
        nu = 1 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2

        res = gpytorch.functions.MaternCovariance.apply(x1, x2, lengthscale, nu, dist_func)
        scaled_unitless_dist = math.sqrt(nu * 2) * dist_func(x1, x2).div(lengthscale)
        exp_component = torch.exp(-scaled_unitless_dist)
        actual = exp_component
        self.assertTrue(torch.allclose(res, actual))

    def test_1_2_backward(self):
        nu = 1 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.MaternCovariance.apply(x1, x2, l, nu, dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")

    def test_3_2_forward(self):
        nu = 3 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2

        res = gpytorch.functions.MaternCovariance.apply(x1, x2, lengthscale, nu, dist_func)
        scaled_unitless_dist = math.sqrt(nu * 2) * dist_func(x1, x2).div(lengthscale)
        exp_component = torch.exp(-scaled_unitless_dist)
        actual = exp_component * (1 + scaled_unitless_dist)
        self.assertTrue(torch.allclose(res, actual))

    def test_3_2_backward(self):
        nu = 3 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.MaternCovariance.apply(x1, x2, l, nu, dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")

    def test_5_2_forward(self):
        nu = 5 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2

        res = gpytorch.functions.MaternCovariance.apply(x1, x2, lengthscale, nu, dist_func)
        scaled_unitless_dist = math.sqrt(nu * 2) * dist_func(x1, x2).div(lengthscale)
        exp_component = torch.exp(-scaled_unitless_dist)
        actual = exp_component * (1 + scaled_unitless_dist + scaled_unitless_dist**2 / 3)
        self.assertTrue(torch.allclose(res, actual))

    def test_5_2_backward(self):
        nu = 5 / 2
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.MaternCovariance.apply(x1, x2, l, nu, dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")


if __name__ == "__main__":
    unittest.main()
