from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from .kernel import Kernel


class RBFKernel(Kernel):
    """
    Computes a covariance matrix based on the RBF (squared exponential) kernel
    between inputs :math:`x_1` and :math:`x_2`:

    .. math::
        \begin{equation*}
            K_{\text{RBF}} = \exp \left( -\frac{1}{2} (x_1 - x_2)^\top \Theta^{-1} (x_1 - x_2) \right)
        \end{equation*}

    where :math:`\Theta` is a `lengthscale` parameter.
    There are a few options for the `lengthscale`:
    - Default: One lengthscale can be applied to all input dimensions/batches
    (i.e. :math:`\Theta` is a constant diagonal matrix).
    - ARD: Each input dimension gets its own separate lengthscale
    (i.e. :math:`\Theta` is a non-constant diagonal matrix).
    This is controlled by the `ard_num_dims` keyword argument.

    In batch-mode (i.e. when `x1` and `x2` are batches of input matrices), each
    batch of data can have its own lengthscale parameter by setting the `batch_size`
    keyword argument to the appropriate number of batches.

    ..note::
        The `lengthscale` parameter is parameterized on a log scale to constrain it to be positive.
        You can set a prior on this parameter using the `log_lengthscale_prior` argument, but
        be sure that the prior has `log_transform=True` set.

    This kernel does not have an `outputscale` parameter. To add a scaling parameter,
    decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        ard_num_dims (int, optional): Set this if you want a separate lengthscale for each
            input dimension. It should be `d` if the `x1` is a `n x d` matrix. Default: `None`
        batch_size (int, optional): Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if the `x1` is a `b x n x d` tensor. Default: `1`
        active_dims (tuple of ints, optional): Set this if you want to compute the covariance of only a few input
            dimensions. The ints corresponds to the indices of the dimensions. Default: `None`.
        log_lengthscale_prior (prior, optional): Set this if you want to apply a prior to the lengthscale parameter.
            Default: `None`
        eps (float): The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        lengthscale (Tensor): The lengthscale parameter. Size/shape of parameter depends on the
            `ard_num_dims` and `batch_size` arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.RBFKernel()
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=5)
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.RBFKernel()
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.RBFKernel(batch_size=2)
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    def __init__(
        self,
        ard_num_dims=None,
        log_lengthscale_prior=None,
        eps=1e-6,
        active_dims=None,
        batch_size=1,
        log_lengthscale_bounds=None,
    ):
        super(RBFKernel, self).__init__(
            has_lengthscale=True,
            ard_num_dims=ard_num_dims,
            log_lengthscale_prior=log_lengthscale_prior,
            active_dims=active_dims,
            batch_size=batch_size,
            log_lengthscale_bounds=log_lengthscale_bounds,
        )
        self.eps = eps

    def forward(self, x1, x2):
        lengthscales = self.log_lengthscale.exp().mul(math.sqrt(2)).clamp(self.eps, 1e5)
        diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).div_(lengthscales.unsqueeze(1))
        return diff.pow_(2).sum(-1).mul_(-1).exp_()
