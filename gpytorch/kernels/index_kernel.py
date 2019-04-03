#!/usr/bin/env python3

import torch
from .kernel import Kernel
from ..lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor
from ..constraints import Positive


class IndexKernel(Kernel):
    r"""
    A kernel for discrete indices. Kernel is defined by a lookup table.

    .. math::

        \begin{equation}
            k(i, j) = \left(BB^\top + \text{diag}(\mathbf v) \right)_{i, j}
        \end{equation}

    where :math:`B` is a low-rank matrix, and :math:`\mathbf v` is a  non-negative vector.
    These parameters are learned.

    Args:
        :attr:`num_tasks` (int):
            Total number of indices.
        :attr:`batch_shape` (torch.Size, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)
        :attr:`rank` (int):
            Rank of :math:`B` matrix.
        :attr:`prior` (:obj:`gpytorch.priors.Prior`):
            Prior for :math:`B` matrix.
        :attr:`var_constraint` (Constraint, optional):
            Constraint for added diagonal component. Default: `Positive`.

    Attributes:
        covar_factor:
            The :math:`B` matrix.
        lov_var:
            The element-wise log of the :math:`\mathbf v` vector.
    """

    def __init__(
        self,
        num_tasks,
        rank=1,
        prior=None,
        var_constraint=None,
        **kwargs
    ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)

        if var_constraint is None:
            var_constraint = Positive()

        self.register_parameter(
            name="covar_factor", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks, rank))
        )
        self.register_parameter(name="raw_var", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks)))
        if prior is not None:
            self.register_prior("IndexKernelPrior", prior, self._eval_covar_matrix)

        self.register_constraint("raw_var", var_constraint)

    @property
    def var(self):
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value):
        self._set_var(value)

    def _set_var(self, value):
        self.initialize(raw_var=self.raw_var_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        var = self.var
        D = var * torch.eye(var.shape[-1], dtype=var.dtype, device=var.device)
        return self.covar_factor.matmul(self.covar_factor.transpose(-1, -2)) + D

    @property
    def covar_matrix(self):
        var = self.var
        res = PsdSumLazyTensor(RootLazyTensor(self.covar_factor), DiagLazyTensor(var))
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = self._eval_covar_matrix()
        res = InterpolatedLazyTensor(base_lazy_tensor=covar_matrix, left_interp_indices=i1, right_interp_indices=i2)
        return res
