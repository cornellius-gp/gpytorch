#!/usr/bin/env python3

from __future__ import annotations

import math

import torch
from linear_operator.operators import (
    LinearOperator,
    LowRankRootLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
)
from torch import nn, Tensor

from ..constraints import GreaterThan, Interval
from ..models import exact_prediction_strategies
from ..priors import LogNormalPrior, Prior
from .kernel import Kernel


def project_onto_unit_sphere(x: Tensor) -> Tensor:
    r"""Inverse stereographic projection"""
    x_sq_norm = x.square().sum(dim=-1, keepdim=True)
    return torch.cat([2 * x, x_sq_norm - 1.0], dim=-1).mul(1.0 / (1.0 + x_sq_norm))


class SphericalLinearKernel(Kernel):
    r"""
    Computes a covariance matrix based on a linear kernel applied after
    inverse stereographic projection:

    .. math::
        k(\mathbf{x_1}, \mathbf{x_2}) = b_0 + b_1
        P(z(\mathbf{x_1}))^\top P(z(\mathbf{x_2}))

    where :math:`z(\mathbf x)` applies lengthscale scaling, :math:`P` is the inverse
    stereographic projection onto a unit sphere, and :math:`(b_0, b_1)` are learned
    mixture weights (via softmax, so :math:`b_0 + b_1 = 1`).

    This kernel was proposed in `We Still Don't Understand High-Dimensional Bayesian Optimization`.
    See https://arxiv.org/abs/2512.00170 for more details.

    Example:
        >>> bounds = torch.stack([torch.zeros(3), torch.ones(3)])  # (2, D) lower and upper
        >>> covar_module = gpytorch.kernels.SphericalLinearKernel(bounds=bounds, ard_num_dims=3)
        >>> x = torch.rand(50, 3)  # data within [0, 1]^3
        >>> covar_matrix = covar_module(x).to_dense()

    :param bounds: Input space bounds, shape `(2, D)` with lower and upper per dimension.
        Used for centering and computing the global lengthscale.
    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if :math:`\mathbf{x_1}` is a `n x d` matrix. (Default: `None`.)
    :param normalize_lengthscale: If True, constrain the ARD lengthscale vector to unit
        L2 norm, thereby speeding up the optimization of hyperparameters. (Default: `False`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: ``LogNormalPrior(loc=sqrt(2), scale=sqrt(3))``.)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: ``GreaterThan(0.025)``.)
    """

    has_lengthscale = True

    def __init__(
        self,
        bounds: Tensor,
        ard_num_dims: int | None = None,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        normalize_lengthscale: bool = False,
        **kwargs,
    ) -> None:
        # Prior similar to Vanilla BO, but without dimensionality scaling (due to global lengthscale)
        if lengthscale_prior is None:
            lengthscale_prior = LogNormalPrior(
                loc=math.sqrt(2),
                scale=math.sqrt(3),
            )
        if lengthscale_constraint is None:
            initial_value = lengthscale_prior.mode if isinstance(lengthscale_prior, Prior) else None
            lengthscale_constraint = GreaterThan(0.025, transform=None, initial_value=initial_value)

        super().__init__(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            **kwargs,
        )
        self.normalize_lengthscale = normalize_lengthscale
        self.bounds = bounds

        # Learned mixture coefficients: softmax([raw_coeffs]) -> [constant, linear]
        self.raw_coeffs = nn.Parameter(torch.zeros(*self.batch_shape, 2))

        # Global lengthscale: sigmoid(raw_glob_ls) * max_sq_norm
        self.raw_glob_ls = nn.Parameter(torch.zeros(*self.batch_shape, 1))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor | LinearOperator:
        if diag:
            # The diagonal is always 1
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)

        if self.normalize_lengthscale:  # Enforce L2 norm = 1
            lengthscale = torch.softmax(self.lengthscale, dim=-1).sqrt()
        else:
            lengthscale = self.lengthscale

        bounds = self.bounds.to(dtype=x1.dtype, device=x1.device)

        # Global lengthscale based on max possible squared norm
        mins, maxs = bounds[0], bounds[1]
        centers = (mins + maxs) / 2.0
        max_sq_norm = ((maxs - mins) / (2 * lengthscale)).square().sum(dim=-1, keepdim=True)
        glob_ls = torch.sqrt(torch.sigmoid(self.raw_glob_ls[..., None]) * max_sq_norm)

        # Mixture coefficients via softmax
        coeffs = torch.softmax(self.raw_coeffs, dim=-1)
        sqrt_const = torch.sqrt(coeffs[..., 0])
        sqrt_linear = torch.sqrt(coeffs[..., 1])

        def _featurize(x: torch.Tensor) -> torch.Tensor:
            x_ = (x - centers) / (lengthscale * glob_ls)
            x_ = project_onto_unit_sphere(x_)
            return torch.cat(
                [
                    x_ * sqrt_linear[..., None, None],
                    sqrt_const[..., None, None].expand(*x_.shape[:-1], 1),
                ],
                dim=-1,
            )

        x1_ = _featurize(x1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLinearOperator when x1 == x2 for efficiency when composing
            # with other kernels
            n, d = x1.shape[-2:]
            return RootLinearOperator(x1_) if d > n else LowRankRootLinearOperator(x1_)

        x2_ = _featurize(x2)
        return MatmulLinearOperator(x1_, x2_.mT)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        # Allow for fast sampling
        return exact_prediction_strategies.LinearPredictionStrategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )
