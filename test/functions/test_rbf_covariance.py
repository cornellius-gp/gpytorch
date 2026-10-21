#!/usr/bin/env python3

import unittest
from unittest.mock import Mock

import torch

import gpytorch


def sq_dist_func(x1, x2):
    dist_module = gpytorch.kernels.kernel.Distance()
    return dist_module._sq_dist(x1, x2)


class TestRBFCovariance(unittest.TestCase):
    def test_lengthscale_gradgradcheck(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for batch_shape in ((), (2,), (2, 1)):
                for same_inputs in (False, True):
                    with self.subTest(device=device, batch_shape=batch_shape, same_inputs=same_inputs):
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
                            return gpytorch.functions.RBFCovariance.apply(x1, x2, value, sq_dist_func)

                        self.assertTrue(torch.autograd.gradcheck(kernel, (lengthscale,)))
                        self.assertTrue(torch.autograd.gradgradcheck(kernel, (lengthscale,)))

    def test_first_order_uses_cached_derivative(self) -> None:
        x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], dtype=torch.double)
        lengthscale = torch.tensor([[0.31]], dtype=torch.double, requires_grad=True)
        distance = Mock(side_effect=sq_dist_func)
        value = gpytorch.functions.RBFCovariance.apply(x, x, lengthscale, distance)
        torch.autograd.grad(value.sum(), lengthscale)
        self.assertEqual(distance.call_count, 1)

    def test_forward(self):
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9)
        x2 = torch.randn(*batch_size, 6, 9)
        # Doesn't support ARD
        lengthscale = torch.randn(*batch_size).view(*batch_size, 1, 1) ** 2
        res = gpytorch.functions.RBFCovariance.apply(x1, x2, lengthscale, sq_dist_func)
        actual = sq_dist_func(x1, x2).div(-2 * lengthscale**2).exp()
        self.assertTrue(torch.allclose(res, actual))

    def test_backward(self):
        batch_size = (3, 2, 4)
        x1 = torch.randn(*batch_size, 7, 9, dtype=torch.float64)
        x2 = torch.randn(*batch_size, 6, 9, dtype=torch.float64)
        lengthscale = torch.randn(*batch_size, dtype=torch.float64, requires_grad=True).view(*batch_size, 1, 1) ** 2
        f = lambda x1, x2, l: gpytorch.functions.RBFCovariance.apply(x1, x2, l, sq_dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")


if __name__ == "__main__":
    unittest.main()
