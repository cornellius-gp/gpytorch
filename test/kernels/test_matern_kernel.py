#!/usr/bin/env python3

import math
import pickle
import unittest
from unittest.mock import patch

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestMatern25BaseKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return MaternKernel(nu=2.5, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return MaternKernel(nu=2.5, ard_num_dims=num_dims, **kwargs)


class TestMatern05BaseKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        kernel = MaternKernel(nu=0.5, **kwargs)
        kernel.initialize(lengthscale=5.0)
        return kernel

    def create_kernel_ard(self, num_dims, **kwargs):
        kernel = MaternKernel(nu=0.5, ard_num_dims=num_dims, **kwargs)
        kernel.initialize(lengthscale=5.0)
        return kernel


class _ExactMaternModel(ExactGP):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, nu: float) -> None:
        super().__init__(x, y, GaussianLikelihood().to(x))
        self.mean_module = ZeroMean()
        self.covar_module = MaternKernel(nu=nu).to(x)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestMaternKernel(unittest.TestCase, BaseKernelTestCase):
    def test_input_lengthscale_mixed_derivatives_keep_generic_dispatch(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for nu in (0.5, 1.5, 2.5):
                with self.subTest(device=device, nu=nu):
                    x = torch.tensor(
                        [[0.03, 0.2], [0.29, 0.6], [0.58, 0.1], [0.87, 0.8]],
                        dtype=torch.double,
                        device=device,
                        requires_grad=True,
                    )
                    kernel = MaternKernel(nu=nu).to(x)
                    kernel.lengthscale = 0.31
                    raw = kernel.raw_lengthscale
                    # Exclude the coincident-input cusp of Matérn-1/2.
                    weights = 1 - torch.eye(len(x), dtype=x.dtype, device=device)
                    with patch("gpytorch.functions.MaternCovariance.apply", side_effect=AssertionError):
                        actual = (kernel(x).to_dense() * weights).sum()
                    distance = (x.unsqueeze(-2) - x.unsqueeze(-3)).square().sum(-1).clamp_min(1e-30).sqrt()
                    scaled = math.sqrt(2 * nu) * distance / kernel.lengthscale
                    polynomial = 1 if nu == 0.5 else 1 + scaled
                    if nu == 2.5:
                        polynomial = polynomial + scaled.square() / 3
                    expected = (polynomial * (-scaled).exp() * weights).sum()
                    gradients = [
                        torch.autograd.grad(value, (x, raw), create_graph=True) for value in (actual, expected)
                    ]
                    for index in (0, 1):
                        torch.testing.assert_close(gradients[0][index], gradients[1][index], atol=1e-10, rtol=1e-9)
                    direction = torch.arange(1, x.numel() + 1, dtype=x.dtype, device=device).reshape_as(x)
                    hessians = [
                        torch.autograd.grad((gx * direction).sum() + gl.sum(), (x, raw)) for gx, gl in gradients
                    ]
                    for index in (0, 1):
                        torch.testing.assert_close(hessians[0][index], hessians[1][index], atol=1e-9, rtol=1e-8)

    def test_optimized_parameter_hessian_matches_gaussian_law_and_gradient_differences(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for nu in (0.5, 1.5, 2.5):
                for objective in ("covariance", "mll"):
                    with self.subTest(device=device, nu=nu, objective=objective):
                        x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], dtype=torch.double, device=device)
                        y = (4 * x).sin().squeeze(-1)
                        model = _ExactMaternModel(x, y, nu)
                        model.covar_module.lengthscale = 0.31
                        model.likelihood.noise = 0.05
                        raw = model.covar_module.raw_lengthscale

                        def evaluate() -> torch.Tensor:
                            if objective == "covariance":
                                return model.covar_module(x).to_dense().sum()
                            return ExactMarginalLogLikelihood(model.likelihood, model)(model(x), y)

                        distance = (x.unsqueeze(-2) - x.unsqueeze(-3)).square().sum(-1).sqrt()
                        scaled = math.sqrt(2 * nu) * distance / model.covar_module.lengthscale
                        polynomial = 1 if nu == 0.5 else 1 + scaled
                        if nu == 2.5:
                            polynomial = polynomial + scaled.square() / 3
                        covariance = polynomial * (-scaled).exp()
                        expected = covariance.sum()
                        if objective == "mll":
                            expected = torch.distributions.MultivariateNormal(
                                torch.zeros_like(y),
                                covariance_matrix=covariance + torch.diag_embed(model.likelihood.noise.expand_as(y)),
                            ).log_prob(y) / len(y)
                        values = (evaluate(), expected)
                        gradients = [torch.autograd.grad(value, raw, create_graph=True)[0] for value in values]
                        hessians = [torch.autograd.grad(gradient.sum(), raw)[0] for gradient in gradients]
                        torch.testing.assert_close(*values, atol=1e-11, rtol=1e-10)
                        torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-9)
                        torch.testing.assert_close(*hessians, atol=1e-9, rtol=1e-8)
                        original = raw.detach().clone()
                        differences = []
                        step = 1e-5
                        for sign in (-1, 1):
                            with torch.no_grad():
                                raw.copy_(original + sign * step)
                            differences.append(torch.autograd.grad(evaluate(), raw)[0])
                        with torch.no_grad():
                            raw.copy_(original)
                        torch.testing.assert_close(
                            hessians[0], (differences[1] - differences[0]) / (2 * step), atol=2e-7, rtol=2e-7
                        )

    def create_kernel_no_ard(self, **kwargs):
        return MaternKernel(nu=1.5, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return MaternKernel(nu=1.5, ard_num_dims=num_dims, **kwargs)

    def test_forward_nu_1_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=0.5).initialize(lengthscale=lengthscale)
        kernel.eval()

        actual = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).div_(-lengthscale).exp()
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_3_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=1.5).initialize(lengthscale=lengthscale)
        kernel.eval()

        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(math.sqrt(3) / lengthscale)
        actual = (dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_5_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=2.5).initialize(lengthscale=lengthscale)
        kernel.eval()

        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(math.sqrt(5) / lengthscale)
        actual = (dist**2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_ard(self):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        b = torch.tensor([[1, 4], [1, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 1, 2)

        kernel = MaternKernel(nu=2.5, ard_num_dims=2)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        dist = torch.tensor([[1, 1], [2, 2]], dtype=torch.float)
        dist.mul_(math.sqrt(5))
        actual = (dist**2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-3)

        # diag
        res = kernel(a, b).diagonal(dim1=-1, dim2=-2)
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dist = torch.tensor([[[0, 0], [2, 2]], [[1, 1], [0, 0]]], dtype=torch.float)
        dist.mul_(math.sqrt(5))
        actual = (dist**2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=-1, dim2=-2)
        actual = torch.cat([actual[i].diagonal(dim1=-1, dim2=-2).unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 3]], [[2, -1, 2], [2, -1, 0]]], dtype=torch.float)
        b = torch.tensor([[[1, 4, 3]], [[2, -1, 0]]], dtype=torch.float)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        kernel = MaternKernel(nu=2.5, batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        dist = torch.tensor([[[1], [1]], [[2], [0]]], dtype=torch.float).mul_(math.sqrt(5))
        actual = (dist**2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_ard_separate_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 3]], [[2, -1, 2], [2, -1, 0]]], dtype=torch.float)
        b = torch.tensor([[[1, 4, 3]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)

        kernel = MaternKernel(nu=2.5, batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        dist = torch.tensor([[[1, 1], [1, 1]], [[4, 4], [0, 0]]], dtype=torch.float).mul_(math.sqrt(5))
        actual = (dist**2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-3)

        # diag
        res = kernel(a, b).diagonal(dim1=-1, dim2=-2)
        actual = torch.cat([actual[i].diagonal(dim1=-1, dim2=-2).unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dist = torch.tensor(
            [
                [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[4.0, 4.0], [0.0, 0.0]]],
            ]
        )

        dist.mul_(math.sqrt(5))
        dist = dist.view(3, 2, 2, 2).transpose(0, 1)
        actual = (dist**2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=-1, dim2=-2)
        actual = actual.diagonal(dim1=-2, dim2=-1)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def create_kernel_with_prior(self, lengthscale_prior):
        return self.create_kernel_no_ard(lengthscale_prior=lengthscale_prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        self.create_kernel_with_prior(None)
        self.create_kernel_with_prior(NormalPrior(0, 1))
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1)

    def test_pickle_with_prior(self):
        kernel = self.create_kernel_with_prior(NormalPrior(0, 1))
        pickle.loads(pickle.dumps(kernel))  # Should be able to pickle and unpickle with a prior


if __name__ == "__main__":
    unittest.main()
