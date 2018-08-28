from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from . import Kernel
from gpytorch.lazy import DiagLazyVariable, ZeroLazyVariable


class WhiteNoiseKernel(Kernel):
    def __init__(self, variances):
        super(WhiteNoiseKernel, self).__init__()
        self.register_buffer("variances", variances)

        # Enforce that variances are batch_size x data_size x task_size
        # where batch_size and task_size are both 1 in the simplest case.
        if variances.ndimension() == 3:
            self.is_batch = True
            self.variances = variances
        elif variances.ndimension() == 2:
            # By default, assume 2 dimensional data means data_size x task_size, and add a batch_size of 1
            self.is_batch = False
            self.variances = variances.unsqueeze(0)
        else:
            self.is_batch = False
            self.variances = variances.unsqueeze(0).unsqueeze(-1)

    def forward(self, x1, x2):
        if self.training and torch.equal(x1, x2):
            # Reshape into a batch of batch_size diagonal matrices, each of which is
            # (data_size * task_size) x (data_size * task_size)
            return DiagLazyVariable(self.variances.view(self.variances.size(0), -1))
        elif x1.size(-2) == x2.size(-2) and x1.size(-2) == self.variances.size(1) and torch.equal(x1, x2):
            return DiagLazyVariable(self.variances.view(self.variances.size(0), -1))
        else:
            return ZeroLazyVariable(x1.size(-3), x1.size(-2), x2.size(-2))
