import unittest

import torch
from torch import nn

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import GibbsKernel, RBFKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class ConstantLengthscale(nn.Module):
    r"""Constant :math:`\ell(x) = \exp(c)`"""

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.log_value = nn.Parameter(torch.tensor(value).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_value.exp().expand(*x.shape[:-1], 1)


class MLPLengthscale(nn.Module):
    """Small MLP, non-constant lengthscale function."""

    def __init__(self, in_dim: int = 1, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.net(x))


class LinearLogLengthscale(nn.Module):
    """Smooth positive lengthscale with a trainable scalar coefficient."""

    def __init__(self) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.coefficient * x[..., :1]).exp()


def _gaussian_overlap_factors(x1: torch.Tensor, x2: torch.Tensor, lengthscale_fn: nn.Module) -> torch.Tensor:
    """One normalized Gaussian inner product per coordinate, without kernel helpers."""
    l1 = lengthscale_fn(x1)
    l2 = lengthscale_fn(x2).transpose(-1, -2)
    variance_sum = l1.square() + l2.square()
    normalizer = (2 * l1 * l2 / variance_sum).sqrt()
    differences = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    return normalizer.unsqueeze(-1) * (-differences.square() / variance_sum.unsqueeze(-1)).exp()


class TestGibbsKernel(BaseKernelTestCase, unittest.TestCase):
    def create_data_no_batch(self):
        return torch.randn(50, 10)

    def create_kernel_no_ard(self, **kwargs):
        return GibbsKernel(ConstantLengthscale(), **kwargs)

    def setUp(self):
        self.lfn = ConstantLengthscale(value=1.0)
        self.kernel = GibbsKernel(self.lfn)

    def test_diagonal_is_one(self):
        r""":math:`k(x, x) = 1` for all :math:`x`."""
        for lfn in [ConstantLengthscale(), MLPLengthscale(in_dim=2)]:
            kernel = GibbsKernel(lfn)
            x = torch.randn(20, 2)
            K = kernel(x).to_dense()
            self.assertTrue(torch.allclose(K.diagonal(), torch.ones(20), atol=1e-5))

    def test_reduces_to_rbf_with_constant_lengthscale(self):
        r"""With constant :math:`\ell(x) = \ell`, Gibbs reduces to RBF."""
        l = 1.5
        kernel_gibbs = GibbsKernel(ConstantLengthscale(value=l))
        kernel_rbf = RBFKernel()
        kernel_rbf.lengthscale = l

        x1 = torch.randn(8, 1)
        x2 = torch.randn(6, 1)

        K_gibbs = kernel_gibbs(x1, x2).to_dense()
        K_rbf = kernel_rbf(x1, x2).to_dense()

        self.assertTrue(torch.allclose(K_gibbs, K_rbf, atol=1e-5))

    def test_gradient_flows_to_lengthscale_fn(self):
        """Gradients propagate through lengthscale_fn."""
        kernel = GibbsKernel(MLPLengthscale(in_dim=2))
        x = torch.randn(8, 2)
        kernel(x).to_dense().sum().backward()

        for name, param in kernel.lengthscale_fn.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

    def test_multidimensional_gaussian_overlap(self) -> None:
        for dtype in (torch.float32, torch.float64):
            for dimensions in (1, 2, 3):
                for batched in (False, True):
                    with self.subTest(dtype=dtype, dimensions=dimensions, batched=batched):
                        x1 = torch.tensor([[0.1, 0.4, -0.2], [0.7, -0.3, 0.5]], dtype=dtype)[..., :dimensions]
                        x2 = torch.tensor([[-0.2, 0.6, 0.3], [0.3, -0.1, -0.4]], dtype=dtype)[..., :dimensions]
                        if batched:
                            x1 = torch.stack((x1, x1 + 0.2))
                        kernel = GibbsKernel(LinearLogLengthscale()).to(dtype=dtype)
                        expected = _gaussian_overlap_factors(x1, x2, kernel.lengthscale_fn).prod(dim=-1)
                        self.assertAllClose(kernel(x1, x2).to_dense(), expected)

    def test_multidimensional_covariance_is_positive_definite(self) -> None:
        x = torch.tensor(
            [[2.0, 3.0, 0.9], [2.1, 0.8, 0.8], [1.8, 0.4, 0.9], [1.7, 1.1, 0.2], [1.6, 0.6, -1.7], [1.1, 2.1, 2.9]],
            dtype=torch.float64,
        )
        kernel = GibbsKernel(LinearLogLengthscale()).double()
        covariance = kernel(x).to_dense()
        mean = torch.zeros(x.size(-2), dtype=x.dtype)
        actual = MultivariateNormal(mean, covariance).log_prob(mean)
        expected_covariance = _gaussian_overlap_factors(x, x, kernel.lengthscale_fn).prod(dim=-1)
        expected = torch.distributions.MultivariateNormal(mean, expected_covariance).log_prob(mean)
        self.assertAllClose(actual, expected)

    def test_multidimensional_gradients_match_gaussian_overlap(self) -> None:
        x1 = torch.tensor([[0.1, 0.4, -0.2], [0.7, -0.3, 0.5]], dtype=torch.float64, requires_grad=True)
        x2 = torch.tensor([[-0.2, 0.6, 0.3], [0.3, -0.1, -0.4]], dtype=torch.float64, requires_grad=True)
        kernel = GibbsKernel(LinearLogLengthscale()).double()
        variables = (x1, x2, kernel.lengthscale_fn.coefficient)
        actual = kernel(x1, x2).to_dense()
        expected = _gaussian_overlap_factors(x1, x2, kernel.lengthscale_fn).prod(dim=-1)
        actual_gradients = torch.autograd.grad(actual.sum(), variables)
        expected_gradients = torch.autograd.grad(expected.sum(), variables)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            self.assertAllClose(actual_gradient, expected_gradient)

    def test_last_dim_is_batch_retains_one_dimensional_overlap(self) -> None:
        x1 = torch.tensor([[0.1, 0.4, -0.2], [0.7, -0.3, 0.5]], dtype=torch.float64)
        x2 = torch.tensor([[-0.2, 0.6, 0.3], [0.3, -0.1, -0.4]], dtype=torch.float64)
        kernel = GibbsKernel(LinearLogLengthscale()).double()
        expected = _gaussian_overlap_factors(x1, x2, kernel.lengthscale_fn).movedim(-1, -3)
        self.assertAllClose(kernel(x1, x2, last_dim_is_batch=True).to_dense(), expected)

    def test_active_dims_use_selected_dimension_count(self) -> None:
        x1 = torch.tensor([[0.1, 0.4, -0.2], [0.7, -0.3, 0.5]], dtype=torch.float64)
        x2 = torch.tensor([[-0.2, 0.6, 0.3], [0.3, -0.1, -0.4]], dtype=torch.float64)
        kernel = GibbsKernel(LinearLogLengthscale(), active_dims=(0, 2)).double()
        expected = _gaussian_overlap_factors(x1[:, [0, 2]], x2[:, [0, 2]], kernel.lengthscale_fn).prod(dim=-1)
        self.assertAllClose(kernel(x1, x2).to_dense(), expected)
