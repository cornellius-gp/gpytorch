#!/usr/bin/env python3

import torch

from ..lazy import DiagLazyTensor, KroneckerProductLazyTensor, lazify
from .kernel import Kernel


class ReversibleGPFAComponentKernel(Kernel):
    r"""
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` that is zeros everywhere except :math:`K_{kernel_loc,kernel_loc}` and returns
    :math:`K = K_{TT} \otimes K_{XX}`. as an :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks (T)
    :param int kernel_loc: Task number that this kernel represents.
        See :class:`gpytorch.kernels.IndexKernel` for details.
    :param dict kwargs: Additional arguments to pass to the kernel.
    """

    def __init__(self, data_covar_module, num_tasks, kernel_loc, **kwargs):
        """
        """
        super(ReversibleGPFAComponentKernel, self).__init__(**kwargs)
        task_diag = torch.zeros(num_tasks)
        task_diag[kernel_loc] = 1
        self.task_covar = DiagLazyTensor(task_diag)
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ReversibleGPFAComponentKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))
        res = KroneckerProductLazyTensor(covar_x, covar_i)
        return res.diag() if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks
