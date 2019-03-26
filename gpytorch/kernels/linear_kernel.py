#!/usr/bin/env python3

import torch
import warnings
from .kernel import Kernel
from ..lazy import MatmulLazyTensor, RootLazyTensor
from ..constraints import Positive


class LinearKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Linear kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::
        \begin{equation*}
            k_\text{Linear}(\mathbf{x_1}, \mathbf{x_2}) = v\mathbf{x_1}^\top
            \mathbf{x_2}.
        \end{equation*}

    where

    * :math:`v` is a :attr:`variance` parameter.


    .. note::

        To implement this efficiently, we use a :obj:`gpytorch.lazy.RootLazyTensor` during training and a
        :class:`gpytorch.lazy.MatmulLazyTensor` during test. These lazy tensors represent matrices of the form
        :math:`K = XX^{\top}` and :math:`K = XZ^{\top}`. This makes inference
        efficient because a matrix-vector product :math:`Kv` can be computed as
        :math:`Kv=X(X^{\top}v)`, where the base multiply :math:`Xv` takes only
        :math:`O(nd)` time and space.

    Args:
        :attr:`variance_prior` (:class:`gpytorch.priors.Prior`):
            Prior over the variance parameter (default `None`).
        :attr:`variance_constraint` (Constraint, optional):
            Constraint to place on variance parameter. Default: `Positive`.
        :attr:`active_dims` (list):
            List of data dimensions to operate on.
            `len(active_dims)` should equal `num_dimensions`.
    """

    def __init__(
        self,
        num_dimensions=None,
        offset_prior=None,
        variance_prior=None,
        variance_constraint=Positive(),
        **kwargs
    ):
        super(LinearKernel, self).__init__(**kwargs)
        if num_dimensions is not None:
            warnings.warn(
                "The `num_dimensions` argument is deprecated and no longer used.",
                DeprecationWarning
            )
            self.register_parameter(
                name="offset",
                parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions))
            )
        if offset_prior is not None:
            warnings.warn(
                "The `offset_prior` argument is deprecated and no longer used.",
                DeprecationWarning
            )
        self.register_parameter(
            name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        if variance_prior is not None:
            self.register_prior(
                "variance_prior",
                variance_prior,
                lambda: self.variance,
                lambda v: self._set_variance(v)
            )

        self.register_constraint("variance_constraint", variance_constraint)

    @property
    def variance(self):
        return self.variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(raw_variance=self.variance_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        x1_ = x1 * self.variance.sqrt()
        if batch_dims == (0, 2):
            x1_ = x1_.view(x1_.size(0), x1_.size(1), -1, 1)
            x1_ = x1_.permute(0, 2, 1, 3).contiguous()
            x1_ = x1_.view(-1, x1_.size(-2), x1_.size(-1))

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyTensor(x1_)

        else:
            x2_ = x2 * self.variance.sqrt()
            if batch_dims == (0, 2):
                x2_ = x2_.view(x2_.size(0), x2_.size(1), -1, 1)
                x2_ = x2_.permute(0, 2, 1, 3).contiguous()
                x2_ = x2_.view(-1, x2_.size(-2), x2_.size(-1))

            prod = MatmulLazyTensor(x1_, x2_.transpose(2, 1))

        return prod
