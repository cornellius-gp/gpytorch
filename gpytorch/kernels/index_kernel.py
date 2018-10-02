from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from ..lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor


def _eval_covar_matrix(covar_factor, log_var):
    return covar_factor.matmul(covar_factor.transpose(-1, -2)) + log_var.exp().diag()


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
        num_tasks (int):
            Total number of indices.
        batch_size (int, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)
        rank (int):
            Rank of :math:`B` matrix.
        prior (:obj:`gpytorch.priors.Prior`):
            Prior for :math:`B` matrix.

    Attributes:
        covar_factor:
            The :math:`B` matrix.
        lov_var:
            The element-wise log of the :math:`\mathbf v` vector.
    """

    def __init__(self, num_tasks, rank=1, batch_size=1, prior=None):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super(IndexKernel, self).__init__()
        self.register_parameter(
            name="covar_factor", parameter=torch.nn.Parameter(torch.randn(batch_size, num_tasks, rank))
        )
        self.register_parameter(name="log_var", parameter=torch.nn.Parameter(torch.randn(batch_size, num_tasks)))
        if prior is not None:
            self.register_derived_prior(
                name="IndexKernelPrior",
                prior=prior,
                parameter_names=("covar_factor", "log_var"),
                transform=_eval_covar_matrix,
            )

    @property
    def covar_matrix(self):
        res = PsdSumLazyTensor(RootLazyTensor(self.covar_factor), DiagLazyTensor(self.log_var.exp()))
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = _eval_covar_matrix(self.covar_factor, self.log_var)
        res = InterpolatedLazyTensor(base_lazy_tensor=covar_matrix, left_interp_indices=i1, right_interp_indices=i2)
        return res
