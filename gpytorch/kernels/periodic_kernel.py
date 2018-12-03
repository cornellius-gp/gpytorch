#!/usr/bin/env python3

import math
import torch
from .kernel import Kernel
from ..utils.deprecation import _deprecate_kwarg
from torch.nn.functional import softplus


class PeriodicKernel(Kernel):
    r""" Computes a covariance matrix based on the periodic kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{Periodic}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left(
            \frac{2 \sin^2 \left( \pi \Vert \mathbf{x_1} - \mathbf{x_2} \Vert_1 / p \right) }
            { \ell^2 } \right)
       \end{equation*}

    where

    * :math:`p` is the periord length parameter.
    * :math:`\ell` is a lengthscale parameter.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    .. note::

        This kernel does not have an ARD lengthscale option.

    Args:
        :attr:`batch_size` (int, optional):
            Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`period_length_prior` (Prior, optional):
            Set this if you want to apply a prior to the period length parameter.  Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`param_transform` (function, optional):
            Set this if you want to use something other than softplus to ensure positiveness of parameters.
        :attr:`inv_param_transform` (function, optional):
            Set this to allow setting parameters directly in transformed space and sampling from priors.
            Automatically inferred for common transformations such as torch.exp or torch.nn.functional.softplus.
        :attr:`eps` (float):
            The minimum value that the lengthscale/period length can take
            (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size = `batch_size x 1 x 1`.
        :attr:`period_length` (Tensor):
            The period length parameter. Size = `batch_size x 1 x 1`.

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

    def __init__(
        self,
        active_dims=None,
        batch_size=1,
        lengthscale_prior=None,
        period_length_prior=None,
        param_transform=softplus,
        inv_param_transform=None,
        eps=1e-6,
        **kwargs
    ):
        lengthscale_prior = _deprecate_kwarg(kwargs, "log_lengthscale_prior", "lengthscale_prior", lengthscale_prior)
        period_length_prior = _deprecate_kwarg(
            kwargs, "log_period_length_prior", "period_length_prior", period_length_prior
        )
        super(PeriodicKernel, self).__init__(
            has_lengthscale=True,
            active_dims=active_dims,
            batch_size=batch_size,
            lengthscale_prior=lengthscale_prior,
            param_transform=param_transform,
            inv_param_transform=inv_param_transform,
            eps=eps,
        )
        self.register_parameter(name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(batch_size, 1, 1)))
        if period_length_prior is not None:
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda: self.period_length,
                lambda v: self._set_period_length(v),
            )

    @property
    def period_length(self):
        return self._param_transform(self.raw_period_length).clamp(self.eps, 1e5)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(raw_period_length=self._inv_param_transform(value))

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.period_length)
        x2_ = x2.div(self.period_length)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        diff = torch.sum((x1_ - x2_).abs(), -1)
        res = torch.sin(diff.mul(math.pi)).pow(2).mul(-2 / self.lengthscale).exp_()
        if diff.ndimension() == 2:
            res = res.squeeze(0)
        return res
