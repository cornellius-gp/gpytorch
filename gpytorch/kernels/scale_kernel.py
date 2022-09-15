#!/usr/bin/env python3

from typing import Optional

import torch
from linear_operator.operators import to_dense

from ..constraints import Interval, Positive
from ..priors import Prior
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
        The outputscale parameter is parameterized on a log scale to constrain it to be positive.
        You can set a prior on this parameter using the outputscale_prior argument.

    Args:
        base_kernel (Kernel):
            The base kernel to be scaled.
        batch_shape (int, optional):
            Set this if you want a separate outputscale for each batch of input data. It should be `b`
            if x1 is a `b x n x d` tensor. Default: `torch.Size([])`
        outputscale_prior (Prior, optional): Set this if you want to apply a prior to the outputscale
            parameter.  Default: `None`
        outputscale_constraint (Constraint, optional): Set this if you want to apply a constraint to the
            outputscale parameter. Default: `Positive`.

    Attributes:
        base_kernel (Kernel):
            The kernel module to be scaled.
        outputscale (Tensor):
            The outputscale parameter. Size/shape of parameter depends on the batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> base_covar_module = gpytorch.kernels.RBFKernel()
        >>> scaled_covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)
        >>> covar = scaled_covar_module(x)  # Output: LinearOperator of size (10 x 10)
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(
        self,
        base_kernel: Kernel,
        outputscale_prior: Optional[Prior] = None,
        outputscale_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(ScaleKernel, self).__init__(**kwargs)
        if outputscale_constraint is None:
            outputscale_constraint = Positive()

        self.base_kernel = base_kernel
        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        if outputscale_prior is not None:
            if not isinstance(outputscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(outputscale_prior).__name__)
            self.register_prior(
                "outputscale_prior", outputscale_prior, self._outputscale_param, self._outputscale_closure
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        return m._set_outputscale(v)

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
            return to_dense(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)
