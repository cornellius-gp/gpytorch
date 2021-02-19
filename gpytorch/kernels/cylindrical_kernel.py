#!/usr/bin/env python3

from typing import Optional

import torch

from .. import settings
from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel


class CylindricalKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Cylindrical Kernel between
    inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.
    It was proposed in `BOCK: Bayesian Optimization with Cylindrical Kernels`.
    See http://proceedings.mlr.press/v80/oh18a.html for more details

    .. note::
        The data must lie completely within the unit ball.

    Args:
        :attr:`num_angular_weights` (int):
            The number of components in the angular kernel
        :attr:`radial_base_kernel` (gpytorch.kernel):
            The base kernel for computing the radial kernel
        :attr:`batch_size` (int, optional):
            Set this if the data is batch of input data.
            It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`
        :attr:`eps` (float):
            Small floating point number used to improve numerical stability
            in kernel computations. Default: `1e-6`
        :attr:`param_transform` (function, optional):
            Set this if you want to use something other than softplus to ensure positiveness of parameters.
        :attr:`inv_param_transform` (function, optional):
            Set this to allow setting parameters directly in transformed space and sampling from priors.
            Automatically inferred for common transformations such as torch.exp or torch.nn.functional.softplus.
    """

    def __init__(
        self,
        num_angular_weights: int,
        radial_base_kernel: Kernel,
        eps: Optional[int] = 1e-6,
        angular_weights_prior: Optional[Prior] = None,
        angular_weights_constraint: Optional[Interval] = None,
        alpha_prior: Optional[Prior] = None,
        alpha_constraint: Optional[Interval] = None,
        beta_prior: Optional[Prior] = None,
        beta_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        if angular_weights_constraint is None:
            angular_weights_constraint = Positive()

        if alpha_constraint is None:
            alpha_constraint = Positive()

        if beta_constraint is None:
            beta_constraint = Positive()

        super().__init__(**kwargs)
        self.num_angular_weights = num_angular_weights
        self.radial_base_kernel = radial_base_kernel
        self.eps = eps

        self.register_parameter(
            name="raw_angular_weights",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, num_angular_weights)),
        )
        self.register_constraint("raw_angular_weights", angular_weights_constraint)
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_alpha", alpha_constraint)
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_beta", beta_constraint)

        if angular_weights_prior is not None:
            self.register_prior(
                "angular_weights_prior",
                angular_weights_prior,
                lambda m: m.angular_weights,
                lambda m, v: m._set_angular_weights(v),
            )
        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, lambda m: m.alpha, lambda m, v: m._set_alpha(v))
        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda m: m.beta, lambda m, v: m._set_beta(v))

    @property
    def angular_weights(self) -> torch.Tensor:
        return self.raw_angular_weights_constraint.transform(self.raw_angular_weights)

    @angular_weights.setter
    def angular_weights(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        self.initialize(raw_angular_weights=self.raw_angular_weights_constraint.inverse_transform(value))

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self) -> torch.Tensor:
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: Optional[bool] = False, **params) -> torch.Tensor:

        x1_, x2_ = x1.clone(), x2.clone()
        # Jitter datapoints that are exactly 0
        x1_[x1_ == 0], x2_[x2_ == 0] = x1_[x1_ == 0] + self.eps, x2_[x2_ == 0] + self.eps
        r1, r2 = x1_.norm(dim=-1, keepdim=True), x2_.norm(dim=-1, keepdim=True)

        if torch.any(r1 > 1.0) or torch.any(r2 > 1.0):
            raise RuntimeError("Cylindrical kernel not defined for data points with radius > 1. Scale your data!")

        a1, a2 = x1.div(r1), x2.div(r2)
        if not diag:
            gram_mat = a1.matmul(a2.transpose(-2, -1))
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None, None].mul(gram_mat.pow(p))
        else:
            gram_mat = a1.mul(a2).sum(-1)
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None].mul(gram_mat.pow(p))

        with settings.lazily_evaluate_kernels(False):
            radial_kernel = self.radial_base_kernel(self.kuma(r1), self.kuma(r2), diag=diag, **params)
        return radial_kernel.mul(angular_kernel)

    def kuma(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(*self.batch_shape, 1, 1)
        beta = self.beta.view(*self.batch_shape, 1, 1)

        res = 1 - (1 - x.pow(alpha) + self.eps).pow(beta)
        return res

    def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
        return self.radial_base_kernel.num_outputs_per_input(x1, x2)
