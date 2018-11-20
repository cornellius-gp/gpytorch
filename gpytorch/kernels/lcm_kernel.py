#!/usr/bin/env python3

from torch.nn import ModuleList
from .kernel import Kernel
from .multitask_kernel import MultitaskKernel


class LCMKernel(Kernel):
    """
    This kernel supports the LCM kernel. It allows the user to specify a list of
    base kernels to use, and individual `MultitaskKernel` objects are fit to each
    of them. The final kernel is the linear sum of the Kronecker product of all
    these base kernels with their respective `MultitaskKernel` objects.

    The returned object is of type :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.
    """

    def __init__(self, base_kernels, num_tasks, rank=1, task_covar_prior=None):
        """
        Args:
            base_kernels (:type: list of `Kernel` objects): A list of base kernels.
            num_tasks (int): The number of output tasks to fit.
            rank (int): Rank of index kernel to use for task covariance matrix for each
                        of the base kernels.
            task_covar_prior (:obj:`gpytorch.priors.Prior`): Prior to use for each
                task kernel. See :class:`gpytorch.kernels.IndexKernel` for details.
        """
        if len(base_kernels) < 1:
            raise ValueError("At least one base kernel must be provided.")
        for k in base_kernels:
            if not isinstance(k, Kernel):
                raise ValueError("base_kernels must only contain Kernel objects")
        super(LCMKernel, self).__init__()
        self.covar_module_list = ModuleList(
            [
                MultitaskKernel(base_kernel, num_tasks=num_tasks, rank=rank, task_covar_prior=task_covar_prior)
                for base_kernel in base_kernels
            ]
        )

    def forward(self, x1, x2, **params):
        res = self.covar_module_list[0].forward(x1, x2, **params)
        for m in self.covar_module_list[1:]:
            res += m.forward(x1, x2, **params)
        return res

    def size(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask kernel
        returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.covar_module_list[0].size(x1, x2)
