#!/usr/bin/env python3

import torch

from ..constraints import Positive
from ..lazy import delazify
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
    batch of data can have its own `outputscale` parameter by setting the `batch_shape`
    keyword argument to the appropriate number of batches.

    .. note::
        The :attr:`outputscale` parameter is parameterized on a log scale to constrain it to be positive.
        You can set a prior on this parameter using the :attr:`outputscale_prior` argument.

    Args:
        :attr:`base_kernel` (Kernel):
            The base kernel to be scaled.
        :attr:`batch_shape` (int, optional):
            Set this if you want a separate outputscale for each batch of input data. It should be `b`
            if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`
        :attr:`outputscale_prior` (Prior, optional): Set this if you want to apply a prior to the outputscale
            parameter.  Default: `None`
        :attr:`outputscale_constraint` (Constraint, optional): Set this if you want to apply a constraint to the
            outputscale parameter. Default: `Positive`.

    Attributes:
        :attr:`base_kernel` (Kernel):
            The kernel module to be scaled.
        :attr:`outputscale` (Tensor):
            The outputscale parameter. Size/shape of parameter depends on the :attr:`batch_shape` arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> base_covar_module = gpytorch.kernels.RBFKernel()
        >>> scaled_covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)
        >>> covar = scaled_covar_module(x)  # Output: LazyTensor of size (10 x 10)
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel, outputscale_prior=None, outputscale_constraint=None, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(ScaleKernel, self).__init__(**kwargs)
        if outputscale_constraint is None:
            outputscale_constraint = Positive()

        self.base_kernel = base_kernel
        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", outputscale_prior, lambda m: m.outputscale, lambda m, v: m._set_outputscale(v)
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        outputscales = self.outputscale
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return delazify(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)
