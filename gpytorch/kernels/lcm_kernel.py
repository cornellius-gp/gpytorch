from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .kernel import Kernel
from .multitask_kernel import MultitaskKernel


class LCMKernel(Kernel):
    """
    This kernel supports the LCM kernel.
    """
    def __init__(self, base_kernel_list, n_tasks, rank=1, task_covar_prior=None):
        """
        Args:
        """
        if len(base_kernel_list) < 1:
            raise ValueError('At least one base kernel must be provided.')
        super(LCMKernel, self).__init__()
        self.base_kernel_list = base_kernel_list
        self.lcm_size = len(base_kernel_list)
        self.covar_module_list = [None] * self.lcm_size
        for i in range(self.lcm_size):
            self.covar_module_list[i] = MultitaskKernel(self.base_kernel_list[i],
                                                        n_tasks=n_tasks, rank=1,
                                                        task_covar_prior=task_covar_prior)

    def forward_diag(self, x1, x2):
        """
        Args:
        """
        res = self.covar_module_list[0].forward_diag(x1, x2)
        for i in range(1, self.lcm_size):
            res += self.covar_module_list[i].forward_diag(x1, x2)
        return res

    def forward(self, x1, x2):
        """
        Args:
        """
        covar_x = self.covar_module_list[0](x1, x2)
        for i in range(1, self.lcm_size):
            covar_x += self.covar_module_list[i](x1, x2)
        return covar_x
