from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
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

    def __init__(self, active_dims=None, batch_size=1, log_period_length_prior=None, eps=1e-6):
        super(CosineKernel, self).__init__(has_lengthscale=False, active_dims=active_dims)
        self.eps = eps
        self.register_parameter(
            name="log_period_length",
            parameter=torch.nn.Parameter(torch.zeros(batch_size, 1, 1)),
            prior=log_period_length_prior,
        )

    @property
    def period_length(self):
        return self.log_period_length.exp().clamp(self.eps, 1e5)

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.period_length)
        x2_ = x2.div(self.period_length)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        diff = torch.norm((x1_ - x2_).abs(), 2, -1)
        res = torch.cos(diff.mul(math.pi))
        return res
