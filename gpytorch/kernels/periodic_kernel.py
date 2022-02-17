#!/usr/bin/env python3

import math
from typing import Optional

import torch

from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel


class PeriodicKernel(Kernel):
    r"""Computes a covariance matrix based on the periodic kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

        \begin{equation*}
            k_{\text{Periodic}}(\mathbf{x}, \mathbf{x'}) = \exp \left(
            -2 \sum_i
            \frac{\sin ^2 \left( \frac{\pi}{p} ({x_{i}} - {x_{i}'} ) \right)}{\lambda}
            \right)
        \end{equation*}

    where

    * :math:`p` is the period length parameter.
    * :math:`\lambda` is a lengthscale parameter.

    Equation is based on `David Mackay's Introduction to Gaussian Processes equation 47`_
    (albeit without feature-specific lengthscales and period lengths). The exponential
    coefficient was changed and lengthscale is not squared to maintain backwards compatibility

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    .. note::

        This kernel does not have an ARD lengthscale or period length option.

    :param batch_shape: (Default: `None`) Set this if you want a separate lengthscale for each
         batch of input data. It should be `torch.Size([b1, b2])` for a `b1 x b2 x n x m` kernel output.
    :type batch_shape: torch.Size, optional
    :param active_dims: (Default: `None`) Set this if you want to
        compute the covariance of only a few input dimensions. The ints
        corresponds to the indices of the dimensions.
    :type active_dims: Tuple(int)
    :param period_length_prior: (Default: `None`)
        Set this if you want to apply a prior to the period length parameter.
    :type period_length_prior: ~gpytorch.priors.Prior, optional
    :param period_length_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the period length parameter.
    :type period_length_constraint: ~gpytorch.constraints.Interval, optional
    :param lengthscale_prior: (Default: `None`)
        Set this if you want to apply a prior to the lengthscale parameter.
    :type lengthscale_prior: ~gpytorch.priors.Prior, optional
    :param lengthscale_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the lengthscale parameter.
    :type lengthscale_constraint: ~gpytorch.constraints.Interval, optional
    :param eps: (Default: 1e-6) The minimum value that the lengthscale can take (prevents divide by zero errors).
    :type eps: float, optional

    :var torch.Tensor lengthscale: The lengthscale parameter. Size = `*batch_shape x 1 x 1`.
    :var torch.Tensor period_length: The period length parameter. Size = `*batch_shape x 1 x 1`.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(batch_size=2))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)

    .. _David Mackay's Introduction to Gaussian Processes equation 47:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.1927&rep=rep1&type=pdf
    """

    has_lengthscale = True

    def __init__(
        self,
        period_length_prior: Optional[Prior] = None,
        period_length_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(PeriodicKernel, self).__init__(**kwargs)
        if period_length_constraint is None:
            period_length_constraint = Positive()

        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if period_length_prior is not None:
            if not isinstance(period_length_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_length_prior).__name__)
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda m: m.period_length,
                lambda m, v: m._set_period_length(v),
            )

        self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # Pop this argument so that we can manually sum over dimensions
        last_dim_is_batch = params.pop("last_dim_is_batch", False)
        # Get lengthscale
        lengthscale = self.lengthscale

        x1_ = x1.div(self.period_length / math.pi)
        x2_ = x2.div(self.period_length / math.pi)
        diff = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True, **params)

        sin_sq = diff.sin().pow(2.0)
        if last_dim_is_batch:
            lengthscale = lengthscale[..., None, :, :]
        else:
            sin_sq = sin_sq.sum(dim=-3)
        if diag:
            lengthscale = lengthscale.squeeze(-1)

        res = sin_sq.div(lengthscale).mul(-2.0).exp()
        return res
