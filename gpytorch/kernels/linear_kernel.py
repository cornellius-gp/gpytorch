#!/usr/bin/env python3

import warnings
from typing import Optional, Union

import torch
from linear_operator.operators import LinearOperator, MatmulLinearOperator, RootLinearOperator
from torch import Tensor

from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel


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

    * :math:`v` is a variance parameter.


    .. note::

        To implement this efficiently, we use a
        :obj:`~linear_operator.operators.RootLinearOperator` during training
        and a :class:`~linear_operator.operators.MatmulLinearOperator` during
        test. These lazy tensors represent matrices of the form :math:`\mathbf
        K = \mathbf X \mathbf X^{\prime \top}`. This makes inference efficient
        because a matrix-vector product :math:`\mathbf K \mathbf v` can be
        computed as :math:`\mathbf K \mathbf v = \mathbf X( \mathbf X^{\prime
        \top} \mathbf v)`, where the base multiply :math:`\mathbf X \mathbf v`
        takes only :math:`\mathcal O(ND)` time and space.

    :param variance_prior: Prior over the variance parameter. (Default `None`.)
    :param variance_constraint: Constraint to place on variance parameter. (Default: `Positive`.)
    :param active_dims: List of data dimensions to operate on. `len(active_dims)` should equal `num_dimensions`.
    """

    def __init__(
        self,
        num_dimensions: Optional[int] = None,
        offset_prior: Optional[Prior] = None,
        variance_prior: Optional[Prior] = None,
        variance_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(LinearKernel, self).__init__(**kwargs)
        if variance_constraint is None:
            variance_constraint = Positive()

        if num_dimensions is not None:
            # Remove after 1.0
            warnings.warn("The `num_dimensions` argument is deprecated and no longer used.", DeprecationWarning)
            self.register_parameter(name="offset", parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions)))
        if offset_prior is not None:
            # Remove after 1.0
            warnings.warn("The `offset_prior` argument is deprecated and no longer used.", DeprecationWarning)
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        if variance_prior is not None:
            if not isinstance(variance_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(variance_prior).__name__)
            self.register_prior("variance_prior", variance_prior, lambda m: m.variance, lambda m, v: m._set_variance(v))

        self.register_constraint("raw_variance", variance_constraint)

    @property
    def variance(self) -> Tensor:
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value: Union[float, Tensor]):
        self._set_variance(value)

    def _set_variance(self, value: Union[float, Tensor]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(
        self, x1: Tensor, x2: Tensor, diag: Optional[bool] = False, last_dim_is_batch: Optional[bool] = False, **params
    ) -> LinearOperator:
        x1_ = x1 * self.variance.sqrt()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLinearOperator when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLinearOperator(x1_)

        else:
            x2_ = x2 * self.variance.sqrt()
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

            prod = MatmulLinearOperator(x1_, x2_.transpose(-2, -1))

        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod
