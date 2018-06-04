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

    def forward(self, x1, x2):
        if self.training:
            return DiagLazyVariable(self.variances.unsqueeze(0))
        elif x1.size(-2) == x2.size(-2) and x1.size(-2) == self.variances.size(-1) and torch.equal(x1, x2):
            return DiagLazyVariable(self.variances.unsqueeze(0))
        else:
            return ZeroLazyVariable(x1.size(-3), x1.size(-2), x2.size(-2))
