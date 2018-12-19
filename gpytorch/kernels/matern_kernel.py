#!/usr/bin/env python3

import math
import torch
from .kernel import Kernel
from ..utils.deprecation import _deprecate_kwarg
from torch.nn.functional import softplus


class MaternKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Matern kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{Matern}}(\mathbf{x_1}, \mathbf{x_2}) = \frac{2^{1 - \nu}}{\Gamma(\nu)}
          \left( \sqrt{2 \nu} d \right) K_\nu \left( \sqrt{2 \nu} d \right)
       \end{equation*}

    where

    * :math:`d = (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-1} (\mathbf{x_1} - \mathbf{x_2})`
      is the distance between
      :math:`x_1` and :math:`x_2` scaled by the :attr:`lengthscale` parameter :math:`\Theta`.
    * :math:`\nu` is a smoothness parameter (takes values 1/2, 3/2, or 5/2). Smaller values are less smooth.
    * :math:`K_\nu` is a modified Bessel function.

    There are a few options for the lengthscale parameter :math:`\Theta`:
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        :attr:`nu` (float):
            The smoothness parameter: either 1/2, 3/2, or 5/2.
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each
            input dimension. It should be `d` if :attr:`x1` is a `n x d` matrix. Default: `None`
        :attr:`batch_size` (int, optional):
            Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to
            compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`log_lengthscale_prior` (Prior, optional):
            Set this if you want
            to apply a prior to the lengthscale parameter.  Default: `None`
        :attr:`param_transform` (function, optional):
            Set this if you want to use something other than softplus to ensure positiveness of parameters.
        :attr:`inv_param_transform` (function, optional):
            Set this to allow setting parameters directly in transformed space and sampling from priors.
            Automatically inferred for common transformations such as torch.exp or torch.nn.functional.softplus.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_size` arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, batch_size=2))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    def __init__(
        self,
        nu=2.5,
        ard_num_dims=None,
        batch_size=1,
        active_dims=None,
        lengthscale_prior=None,
        param_transform=softplus,
        inv_param_transform=None,
        eps=1e-6,
        **kwargs
    ):
        _deprecate_kwarg(kwargs, "log_lengthscale_prior", "lengthscale_prior", lengthscale_prior)
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel, self).__init__(
            has_lengthscale=True,
            ard_num_dims=ard_num_dims,
            batch_size=batch_size,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            param_transform=param_transform,
            inv_param_transform=inv_param_transform,
            eps=eps,
        )
        self.nu = nu

    def forward(self, x1, x2, **params):
        mean = x1.contiguous().view(-1, 1, x1.size(-1)).mean(0, keepdim=True)

        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        distance = (x1_ - x2_).norm(2, dim=-1)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        return constant_component * exp_component
