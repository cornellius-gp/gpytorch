import unittest

import torch
from torch import nn

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
