from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel


class RQKernel(Kernel):

    r"""
    Computes a covariance matrix based on the rational quadratic kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{RQ}}(\mathbf{x_1}, \mathbf{x_2}) =  \left( 1 + \frac{||x_1 - x_2||^2}{2 \alpha \ell^2}
          \right)^{-\alpha}
       \end{equation*}

    where :math:`\ell` is a :attr:`lengthscale` parameter, and :math:`\alpha` is a parameter of
    a gamma prior on :math:`\ell^{-2}`.
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.
        This kernel does not support an :attr:`ard_num_dims` argument.

    Args:
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
        :attr:`log_alpha_prior` (Prior, optional):
            Set this if you want
            to apply a prior to the alpha parameter.  Default: `None`
        :attr:`eps` (float):
            The minimum value that the lengthscale can take
            (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`batch_size` argument.
        :attr:`log_alpha` (Tensor):
            The inverse lengthscale prior parameter. Size/shape of parameter depends on the :attr:`batch_size`
            argument.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch:
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch:
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(batch_size=2))
    """

    def __init__(self, log_lengthscale_prior=None, log_alpha_prior=None,
                 eps=1e-6, active_dims=None, batch_size=1):
        super(RQKernel, self).__init__(
            has_lengthscale=True,
            batch_size=batch_size,
            active_dims=active_dims,
            log_lengthscale_prior=log_lengthscale_prior,
            eps=eps
        )
        self.register_parameter(
            name="log_alpha",
            parameter=torch.nn.Parameter(torch.zeros(batch_size, 1, 1)),
            prior=log_alpha_prior
        )

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)
        alpha = self.log_alpha.exp()

        diff = (x1_ - x2_).norm(2, dim=-1)
        res = (1 + diff.pow(2).div(2 * alpha)).pow(-alpha)
        return res
