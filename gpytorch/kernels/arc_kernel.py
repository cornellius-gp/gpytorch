#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:13:37 2020
@author: kostal
"""

from math import pi
from typing import Optional

import torch

from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior


class ArcKernel(Kernel):

    r""" Computes a covariance matrix based on the Arc Kernel
    (https://arxiv.org/abs/1409.4011) between inputs :math:`\mathbf{x_1}`
    and :math:`\mathbf{x_2}`. First it applies a cylindrical embedding:
    .. math::
          g_{i}(\mathbf{x}) = \{\begin{eqnarray}
          [0, 0]^{T} \qquad if\;\delta_{i}(\mathbf{x}) = false\\
          \omega_{i}[\sin{\pi\rho_{i}\frac{x_{i}}{u_{i}-l_{i}}},
          \cos{\pi\rho_{i}\frac{x_{i}}{u_{i}-l_{i}}}] \qquad otherwise
          \end{eqnarray}
    where
    * :math:`\rho` is the angle parameter.
    * :math:`\omega` is a radius parameter.
    then the kernel is built with the particular covariance function, e.g.
    .. math::
        \begin{equation}
        k_{i}(\mathbf{x}, \mathbf{x^{'}}) =
        \sigma^{2}\exp(-\frac{1}{2}d_{i}(\mathbf{x}, \mathbf{x^{'}}))^{2}
        \end{equation}
    and the produt between dimensions
    .. math::
        \begin{equation}
        k_{i}(\mathbf{x}, \mathbf{x^{'}}) =
        \sigma^{2}\exp(-\frac{1}{2}d_{i}(\mathbf{x}, \mathbf{x^{'}}))^{2}
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
    Args:
        :attr:`base_kernel` (gpytorch.kernels.Kernel):
            The euclidean covariance of choice. Default: `MaternKernel(nu=2.5)`
        :attr:`ard_num_dims` (int):
            The number of dimensions to compute the kernel for. The kernel has
            two parameters which are individually defined for each dimension.
            Default: `None`.
        :attr:`angle_prior` (Prior, optional):
            Set this if you want to apply a prior to the period angle
            parameter.  Default: `None`.
        :attr:`radius_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.
            Default: `None`.
    Attributes:
        :attr:`radius` (Tensor):
            The radius parameter. Size = `*batch_shape  x 1`.
        :attr:`angle` (Tensor):
            The period angle parameter. Size = `*batch_shape  x 1`.
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
        self, base_kernel, angle_prior: Optional[Prior] = None, radius_prior: Optional[Prior] = None, **kwargs
    ):
        super(ArcKernel, self).__init__(has_lengthscale=True, **kwargs)

        if self.ard_num_dims is None:
            last_dim = 1
        else:
            last_dim = self.ard_num_dims
        # TODO: check the errors given by interval
        angle_constraint = Positive()

        self.register_parameter(
            name="raw_angle", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, last_dim)),
        )
        if angle_prior is not None:
            self.register_prior(
                "angle_prior", angle_prior, lambda: self.angle, lambda v: self._set_angle(v),
            )

        self.register_constraint("raw_angle", angle_constraint)

        self.register_parameter(
            name="raw_radius", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, last_dim)),
        )

        if radius_prior is not None:
            self.register_prior(
                "radius_prior", radius_prior, lambda: self.radius, lambda v: self._set_radius(v),
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
        x_ = x.div(self.lengthscale)
        x_s = self.radius * torch.sin(pi * self.angle * x_)
        x_c = self.radius * torch.cos(pi * self.angle * x_)
        x_ = torch.cat((x_s, x_c), dim=-1).squeeze(0)
        return x_

    def forward(self, x1, x2, diag=False, **params):
        x1_, x2_ = self.embedding(x1), self.embedding(x2)
        return self.base_kernel(x1_, x2_)
