#!/usr/bin/env python3

import math

import torch

from ..constraints import Positive
from .kernel import Kernel


class PeriodicKernel(Kernel):
    r"""Computes a covariance matrix based on the periodic kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

        \begin{equation*}
            k_{\text{Periodic}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left(
            -2 \sum_i
            \frac{\sin ^2 \left( \frac{\pi}{p} (\mathbf{x_{1,i}} - \mathbf{x_{2,i}} ) \right)}{\lambda}
            \right)
        \end{equation*}

    where

    * :math:`p` is the period length parameter.
    * :math:`\lambda` is a lengthscale parameter.

    Equation is based on [David Mackay's Introduction to Gaussian Processes equation 47]
    (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.1927&rep=rep1&type=pdf)
    albeit without feature-specific lengthscales and period lengths. The exponential
    coefficient was changed and lengthscale is not squared to maintain backwards compatibility

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    .. note::

        This kernel does not have an ARD lengthscale or period length option.

    Args:
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`period_length_prior` (Prior, optional):
            Set this if you want to apply a prior to the period length parameter.  Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the value of the lengthscale. Default: `Positive`.
        :attr:`period_length_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the value of the period length. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale/period length can take
            (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size = `*batch_shape x 1 x 1`.
        :attr:`period_length` (Tensor):
            The period length parameter. Size = `*batch_shape x 1 x 1`.

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
    """

    has_lengthscale = True

    def __init__(self, period_length_prior=None, period_length_constraint=None, **kwargs):
        super(PeriodicKernel, self).__init__(**kwargs)
        if period_length_constraint is None:
            period_length_constraint = Positive()

        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

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
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.period_length).mul(math.pi)
        x2_ = x2.div(self.period_length).mul(math.pi)
        diff = x1_.unsqueeze(-2) - x2_.unsqueeze(-3)
        res = diff.sin().pow(2).sum(dim=-1).div(self.lengthscale).mul(-2.0).exp_()
        if diag:
            res = res.squeeze(0)
        return res
