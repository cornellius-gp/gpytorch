#!/usr/bin/env python3

from math import pi
from typing import Optional

import torch

from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel


class ArcKernel(Kernel):
    r""" Computes a covariance matrix based on the Arc Kernel
    (https://arxiv.org/abs/1409.4011) between inputs :math:`\mathbf{x_1}`
    and :math:`\mathbf{x_2}`. First it applies a cylindrical embedding:

    .. math::
        g_{i}(\mathbf{x}) = \begin{cases}
        [0, 0]^{T} & \delta_{i}(\mathbf{x}) = \text{false}\\
        \omega_{i} \left[ \sin{\pi\rho_{i}\frac{x_{i}}{u_{i}-l_{i}}},
        \cos{\pi\rho_{i}\frac{x_{i}}{u_{i}-l_{i}}} \right] & \text{otherwise}
        \end{cases}

    where
    * :math:`\rho` is the angle parameter.
    * :math:`\omega` is a radius parameter.

    then the kernel is built with the particular covariance function, e.g.

    .. math::
        \begin{equation}
        k_{i}(\mathbf{x}, \mathbf{x'}) =
        \sigma^{2}\exp \left(-\frac{1}{2}d_{i}(\mathbf{x}, \mathbf{x^{'}}) \right)^{2}
        \end{equation}

    and the produt between dimensions

    .. math::
        \begin{equation}
        k_{i}(\mathbf{x}, \mathbf{x'}) =
        \sigma^{2}\exp \left(-\frac{1}{2}d_{i}(\mathbf{x}, \mathbf{x^{'}}) \right)^{2}
        \end{equation}

    .. note::
        This kernel does not have an `outputscale` parameter. To add a scaling
        parameter, decorate this kernel with a
        :class:`gpytorch.kernels.ScaleKernel`.
        When using with an input of `b x n x d` dimensions, decorate this
        kernel with :class:`gpytorch.kernel.ProductStructuredKernel , setting
        the number of dims, `num_dims to d.`

    .. note::
        This kernel does not have an ARD lengthscale option.

    :param base_kernel: (Default :obj:`gpytorch.kernels.MaternKernel(nu=2.5)`.)
        The euclidean covariance of choice.
    :type base_kernel: :obj:`~gpytorch.kernels.Kernel`
    :param ard_num_dims: (Default `None`.) The number of dimensions to compute the kernel for.
        The kernel has two parameters which are individually defined for each
        dimension, defaults to None
    :type ard_num_dims: int, optional
    :param angle_prior: Set this if you want to apply a prior to the period angle parameter.
    :type angle_prior: :obj:`~gpytorch.priors.Prior`, optional
    :param radius_prior: Set this if you want to apply a prior to the lengthscale parameter.
    :type radius_prior: :obj:`~gpytorch.priors.Prior`, optional

    :var torch.Tensor radius: The radius parameter. Size = `*batch_shape  x 1`.
    :var torch.Tensor angle: The period angle parameter. Size = `*batch_shape  x 1`.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        ... base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        >>> base_kernel.raw_lengthscale.requires_grad_(False)
        >>> covar_module = gpytorch.kernels.ProductStructureKernel(
                gpytorch.kernels.ScaleKernel(
                    ArcKernel(base_kernel,
                              angle_prior=gpytorch.priors.GammaPrior(0.5,1),
                              radius_prior=gpytorch.priors.GammaPrior(3,2),
                              ard_num_dims=x.shape[-1])),
                num_dims=x.shape[-1])
        >>> covar = covar_module(x)
        >>> print(covar.shape)
        >>> # Now with batch
        >>> covar_module = gpytorch.kernels.ProductStructureKernel(
                gpytorch.kernels.ScaleKernel(
                    ArcKernel(base_kernel,
                              angle_prior=gpytorch.priors.GammaPrior(0.5,1),
                              radius_prior=gpytorch.priors.GammaPrior(3,2),
                              ard_num_dims=x.shape[-1])),
                num_dims=x.shape[-1])
        >>> covar = covar_module(x
        >>> print(covar.shape)
    """

    has_lengthscale = True

    def __init__(
        self,
        base_kernel,
        delta_func: Optional = None,
        angle_prior: Optional[Prior] = None,
        radius_prior: Optional[Prior] = None,
        **kwargs,
    ):
        super(ArcKernel, self).__init__(has_lengthscale=True, **kwargs)

        if self.ard_num_dims is None:
            self.last_dim = 1
        else:
            self.last_dim = self.ard_num_dims

        if delta_func is None:
            self.delta_func = self.default_delta_func
        else:
            self.delta_func = delta_func

        # TODO: check the errors given by interval
        angle_constraint = Interval(0.1, 0.9)
        self.register_parameter(
            name="raw_angle", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.last_dim)),
        )
        if angle_prior is not None:
            self.register_prior(
                "angle_prior", angle_prior, lambda m: m.angle, lambda m, v: m._set_angle(v),
            )

        self.register_constraint("raw_angle", angle_constraint)

        self.register_parameter(
            name="raw_radius", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.last_dim)),
        )

        if radius_prior is not None:
            self.register_prior(
                "radius_prior", radius_prior, lambda m: m.radius, lambda m, v: m._set_radius(v),
            )

        radius_constraint = Positive()
        self.register_constraint("raw_radius", radius_constraint)

        self.base_kernel = base_kernel
        if self.base_kernel.has_lengthscale:
            self.base_kernel.lengthscale = 1
            self.base_kernel.raw_lengthscale.requires_grad_(False)

    @property
    def angle(self):
        return self.raw_angle_constraint.transform(self.raw_angle)

    @angle.setter
    def angle(self, value):
        self._set_angle(value)

    def _set_angle(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_angle)
        self.initialize(raw_angle=self.raw_angle_constraint.inverse_transform(value))

    @property
    def radius(self):
        return self.raw_radius_constraint.transform(self.raw_radius)

    @radius.setter
    def radius(self, value):
        self._set_radius(value)

    def _set_radius(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_radius)
        self.initialize(raw_radius=self.raw_radius_constraint.inverse_transform(value))

    def embedding(self, x):
        mask = self.delta_func(x)
        x_ = x.div(self.lengthscale)
        x_s = self.radius * torch.sin(pi * self.angle * x_) * mask
        x_c = self.radius * torch.cos(pi * self.angle * x_) * mask
        x_ = torch.cat((x_s, x_c), dim=-1)
        return x_

    def default_delta_func(self, x):
        return torch.ones_like(x)

    def forward(self, x1, x2, diag=False, **params):
        x1_, x2_ = self.embedding(x1), self.embedding(x2)
        return self.base_kernel(x1_, x2_, diag=diag)
