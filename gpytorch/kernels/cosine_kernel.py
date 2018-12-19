#!/usr/bin/env python3

import math
import torch
from .kernel import Kernel
from ..utils.deprecation import _deprecate_kwarg
from torch.nn.functional import softplus


class CosineKernel(Kernel):
    r"""
    Computes a covariance matrix based on the cosine kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{Cosine}}(\mathbf{x_1}, \mathbf{x_2}) = \cos \left(
            \pi \Vert \mathbf{x_1} - \mathbf{x_2} \Vert_2 / p \right)
       \end{equation*}

    where :math:`p` is the periord length parameter.

    Args:
        :attr:`batch_size` (int, optional):
            Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`log_period_length_prior` (Prior, optional):
            Set this if you want to apply a prior to the period length parameter.  Default: `None`
        :attr:`eps` (float):
            The minimum value that the lengthscale/period length can take
            (prevents divide by zero errors). Default: `1e-6`.
        :attr:`param_transform` (function, optional):
            Set this if you want to use something other than softplus to ensure positiveness of parameters.
        :attr:`inv_param_transform` (function, optional):
            Set this to allow setting parameters directly in transformed space and sampling from priors.
            Automatically inferred for common transformations such as torch.exp or torch.nn.functional.softplus.

    Attributes:
        :attr:`period_length` (Tensor):
            The period length parameter. Size = `batch_size x 1 x 1`.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(batch_size=2))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    def __init__(
        self,
        active_dims=None,
        batch_size=1,
        period_length_prior=None,
        eps=1e-6,
        param_transform=softplus,
        inv_param_transform=None,
        **kwargs
    ):
        period_length_prior = _deprecate_kwarg(
            kwargs, "log_period_length_prior", "period_length_prior", period_length_prior
        )
        super(CosineKernel, self).__init__(
            active_dims=active_dims, param_transform=param_transform, inv_param_transform=inv_param_transform
        )
        self.eps = eps
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
        return self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(raw_period_length=self._inv_param_transform(value))

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.period_length)
        x2_ = x2.div(self.period_length)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        diff = torch.norm((x1_ - x2_).abs(), 2, -1)
        res = torch.cos(diff.mul(math.pi))
        return res
