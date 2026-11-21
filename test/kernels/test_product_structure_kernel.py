#!/usr/bin/env python3

import math
import unittest
import warnings

import torch

from gpytorch.kernels import GridInterpolationKernel, ProductStructureKernel, RBFKernel, ScaleKernel


class TestProductStructureKernel(unittest.TestCase):
    def _make_kernel(self, interpolated: bool = False, num_dims: int = 4) -> ProductStructureKernel:
        base = RBFKernel()
        if interpolated:
            base = ScaleKernel(GridInterpolationKernel(base, grid_size=32, num_dims=1, grid_bounds=[(0.0, 1.0)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return ProductStructureKernel(base, num_dims=num_dims).double()

    def test_rectangular_and_distinct_square_inputs(self) -> None:
        torch.manual_seed(0)
        for interpolated in (False, True):
            for n2 in (5, 9):  # n2 != n1 (rectangular) and n2 == n1 with x1 != x2 (asymmetric square)
                with self.subTest(interpolated=interpolated, n2=n2):
                    kernel = self._make_kernel(interpolated=interpolated, num_dims=4)
                    x1 = torch.rand(9, 4, dtype=torch.double, requires_grad=True)
                    x2 = torch.rand(n2, 4, dtype=torch.double, requires_grad=True)

                    expected = math.prod(
                        kernel.base_kernel(x1[:, d : d + 1], x2[:, d : d + 1]).to_dense() for d in range(4)
                    )
                    actual_call = kernel(x1, x2).to_dense()
                    actual_forward = kernel.forward(x1, x2).to_dense()

                    torch.testing.assert_close(actual_call, expected, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(actual_forward, expected, rtol=1e-10, atol=1e-10)

                    grads_actual = torch.autograd.grad(actual_call.sum(), (x1, x2))
                    grads_expected = torch.autograd.grad(expected.sum(), (x1, x2))
                    for ga, ge in zip(grads_actual, grads_expected, strict=True):
                        torch.testing.assert_close(ga, ge, rtol=1e-9, atol=1e-9)

    def test_diag_evaluation(self) -> None:
        torch.manual_seed(1)
        for batch in ((), (2,)):
            with self.subTest(batch=batch):
                kernel = self._make_kernel(interpolated=True, num_dims=3)
                x1 = torch.rand(*batch, 6, 3, dtype=torch.double)
                x2 = torch.rand(*batch, 6, 3, dtype=torch.double)

                diag_same = kernel(x1, diag=True)
                expected_same = kernel(x1, x1).to_dense().diagonal(dim1=-2, dim2=-1)
                # Note: square x1 == x2 uses root_decomposition approximation; check against exact 1D product
                exact_same = math.prod(kernel.base_kernel(x1[..., d : d + 1], diag=True) for d in range(3))
                torch.testing.assert_close(diag_same, exact_same, rtol=1e-10, atol=1e-10)
                self.assertEqual(diag_same.shape, expected_same.shape)

                diag_pair = kernel(x1, x2, diag=True)
                expected_pair = kernel(x1, x2).to_dense().diagonal(dim1=-2, dim2=-1)
                torch.testing.assert_close(diag_pair, expected_pair, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
