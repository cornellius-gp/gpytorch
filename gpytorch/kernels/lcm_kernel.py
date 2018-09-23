from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.nn import ModuleList
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.multitask_kernel import MultitaskKernel


class LCMKernel(Kernel):
    """
    This kernel supports the LCM kernel.
    """
    def __init__(self, base_kernels, n_tasks, rank=1, task_covar_prior=None):
        """
        Args:
        """
        if len(base_kernels) < 1:
            raise ValueError('At least one base kernel must be provided.')
        for k in base_kernels:
            if not isinstance(k, Kernel):
                raise ValueError("base_kernels must only contain Kernel objects")
        super(LCMKernel, self).__init__()
        self.covar_module_list = ModuleList([
            MultitaskKernel(base_kernel, n_tasks=n_tasks, rank=rank, task_covar_prior=task_covar_prior)
            for base_kernel in base_kernels
        ])

    def forward_diag(self, x1, x2):
        """
        Args:
        """
        res = self.covar_module_list[0].forward_diag(x1, x2)
        for m in self.covar_module_list[1:]:
            res += m.forward_diag(x1, x2)
        return res

    def forward(self, x1, x2):
        """
        Args:
        """
        res = self.covar_module_list[0].forward(x1, x2)
        for m in self.covar_module_list[1:]:
            res += m.forward(x1, x2)
        return res

    def size(self, x1, x2):
        return self.covar_module_list[0].size(x1, x2)
