from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel


class ScaleKernel(Kernel):
    r"""
    Decorates an existing kernel object with an output scale, i.e.

    .. math::

       \begin{equation*}
          K_{\text{scaled}} = \theta_\text{scale} K_{\text{orig}}
       \end{equation*}

    where :math:`\theta_\text{scale}` is the `outputscale` parameter.

    In batch-mode (i.e. when :math:`x_1` and :math:`x_2` are batches of input matrices), each
    batch of data can have its own `outputscale` parameter by setting the `batch_size`
    keyword argument to the appropriate number of batches.

    .. note::
        The :attr:`outputscale` parameter is parameterized on a log scale to constrain it to be positive.
        You can set a prior on this parameter using the :attr:`log_outputscale_prior` argument, but
        be sure that the prior has :attr:`log_transform=True` set.

    Args:
        :attr:`batch_size` (int, optional): Set this if you want a separate outputscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`
        :attr:`log_outputscale_prior` (Prior, optional): Set this if you want
            to apply a prior to the outputscale parameter.  Default: `None`

    Attributes:
        :attr:`base_kernel` (Kernel): The kernel module to be scaled.
        :attr:`outputscale` (Tensor): The outputscale parameter. Size/shape of parameter depends on the
            :attr:`batch_size` arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> base_covar_module = gpytorch.kernels.RBFKernel()
        >>> scaled_covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)
        >>> covar = scaled_covar_module(x)  # Output: LazyTensor of size (10 x 10)
    """

    def __init__(self, base_kernel, batch_size=1, log_outputscale_prior=None):
        super(ScaleKernel, self).__init__(has_lengthscale=False, batch_size=batch_size)
        self.base_kernel = base_kernel
        self.register_parameter(
            name="log_outputscale", parameter=torch.nn.Parameter(torch.zeros(batch_size)), prior=log_outputscale_prior
        )

    @property
    def outputscale(self):
        return self.log_outputscale.exp()

    def forward(self, x1, x2, batch_dims=None, **params):
        outputscales = self.log_outputscale.exp()
        if batch_dims == (0, 2) and outputscales.numel() > 1:
            outputscales = outputscales.unsqueeze(1).repeat(1, x1.size(-1)).view(-1)

        orig_output = self.base_kernel(x1, x2, batch_dims=batch_dims, **params)
        if torch.is_tensor(orig_output):
            outputscales = outputscales.view(-1, *([1] * (orig_output.dim() - 1)))
        return orig_output.mul(outputscales)
