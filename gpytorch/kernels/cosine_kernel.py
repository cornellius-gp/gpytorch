#!/usr/bin/env python3

import math

import torch

from ..constraints import Positive
from .kernel import Kernel


class CosineKernel(Kernel):
    r"""
    Computes a covariance matrix based on the cosine kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{Cosine}}(\mathbf{x_1}, \mathbf{x_2}) = \cos \left(
            \pi \Vert \mathbf{x_1} - \mathbf{x_2} \Vert_2 / p \right)
       \end{equation*}

    where :math:`p` is the period length parameter.

    Args:
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`period_length_prior` (Prior, optional):
            Set this if you want to apply a prior to the period length parameter.  Default: `None`
        :attr:`period_length_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the period length parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale/period length can take
            (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`period_length` (Tensor):
            The period length parameter. Size = `*batch_shape x 1 x 1`.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(batch_shape=torch.Size([2])))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    is_stationary = True

    def __init__(self, period_length_prior=None, period_length_constraint=None, **kwargs):
        super(CosineKernel, self).__init__(**kwargs)

        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if period_length_constraint is None:
            period_length_constraint = Positive()

        if period_length_prior is not None:
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
        return self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)

        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.period_length)
        x2_ = x2.div(self.period_length)
        diff = self.covar_dist(x1_, x2_, **params)
        res = torch.cos(diff.mul(math.pi))
        return res
