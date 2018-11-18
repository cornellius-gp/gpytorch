#!/usr/bin/env python3

import torch
from .kernel import Kernel
from ..lazy import MatmulLazyTensor, RootLazyTensor


class LinearKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Linear kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::
        \begin{equation*}
            k_\text{Linear}(\mathbf{x_1}, \mathbf{x_2}) = (\mathbf{x_1} - \mathbf{o})^\top
            (\mathbf{x_2} - \mathbf{o}) + v.
        \end{equation*}

    where

    * :math:`\mathbf o` is an :attr:`offset` parameter.
    * :math:`v` is a :attr:`variance` parameter.


    .. note::

        To implement this efficiently, we use a :obj:`gpytorch.lazy.RootLazyTensor` during training and a
        :class:`gpytorch.lazy.MatmulLazyTensor` during test. These lazy tensors represent matrices of the form
        :math:`K = XX^{\top}` and :math:`K = XZ^{\top}`. This makes inference
        efficient because a matrix-vector product :math:`Kv` can be computed as
        :math:`Kv=X(X^{\top}v)`, where the base multiply :math:`Xv` takes only
        :math:`O(nd)` time and space.

    Args:
        :attr:`num_dimensions` (int):
            Number of data dimensions to expect. This
            is necessary to create the offset parameter.
        :attr:`variance_prior` (:class:`gpytorch.priors.Prior`):
            Prior over the variance parameter (default `None`).
        :attr:`offset_prior` (:class:`gpytorch.priors.Prior`):
            Prior over the offset parameter (default `None`).
        :attr:`active_dims` (list):
            List of data dimensions to operate on.
            `len(active_dims)` should equal `num_dimensions`.
    """

    def __init__(self, num_dimensions, variance_prior=None, offset_prior=None, active_dims=None):
        super(LinearKernel, self).__init__(active_dims=active_dims)
        self.register_parameter(name="variance", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="offset", parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions)))
        if variance_prior is not None:
            self.register_prior("variance_prior", variance_prior, "variance")
        if offset_prior is not None:
            self.register_prior("offset_prior", offset_prior, "offset")

    def forward(self, x1, x2, batch_dims=None, **params):
        x1_ = x1 - self.offset
        if batch_dims == (0, 2):
            x1_ = x1_.view(x1_.size(0), x1_.size(1), -1, 1)
            x1_ = x1_.permute(0, 2, 1, 3).contiguous()
            x1_ = x1_.view(-1, x1_.size(-2), x1_.size(-1))

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyTensor(x1_)

        else:
            x2_ = x2 - self.offset
            if batch_dims == (0, 2):
                x2_ = x2_.view(x2_.size(0), x2_.size(1), -1, 1)
                x2_ = x2_.permute(0, 2, 1, 3).contiguous()
                x2_ = x2_.view(-1, x2_.size(-2), x2_.size(-1))

            prod = MatmulLazyTensor(x1_, x2_.transpose(2, 1))

        return prod + self.variance.expand(prod.size())
