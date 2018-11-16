from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from ..lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor


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
        :attr:`batch_size` (int, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)
        :attr:`rank` (int):
            Rank of :math:`B` matrix.
        :attr:`prior` (:obj:`gpytorch.priors.Prior`):
            Prior for :math:`B` matrix.
        :attr:`param_transform` (function, optional):
            Set this if you want to use something other than torch.exp to ensure positiveness of parameters.

    Attributes:
        covar_factor:
            The :math:`B` matrix.
        lov_var:
            The element-wise log of the :math:`\mathbf v` vector.
    """

    def __init__(self, num_tasks, rank=1, batch_size=1, prior=None, param_transform=torch.exp):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super(IndexKernel, self).__init__(param_transform=param_transform)
        self.register_parameter(
            name="covar_factor", parameter=torch.nn.Parameter(torch.randn(batch_size, num_tasks, rank))
        )
        self.register_parameter(name="log_var", parameter=torch.nn.Parameter(torch.randn(batch_size, num_tasks)))
        if prior is not None:
            self.register_prior("IndexKernelPrior", prior, self._eval_covar_matrix)

    @property
    def var(self):
        return self._param_transform(self.log_var)

    def _eval_covar_matrix(self):
        covar_factor, variance = self.covar_factor, self.var
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + variance.diag()
        # D = var * torch.eye(var.shape[-1], dtype=var.dytpe, device=var.device)
        # return self.covar_factor.matmul(self.covar_factor.transpose(-1, -2)) + D

    @property
    def covar_matrix(self):
        var = self.var
        res = PsdSumLazyTensor(RootLazyTensor(self.covar_factor), DiagLazyTensor(var))
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = self._eval_covar_matrix()
        res = InterpolatedLazyTensor(base_lazy_tensor=covar_matrix, left_interp_indices=i1, right_interp_indices=i2)
        return res
