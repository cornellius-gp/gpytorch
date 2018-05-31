import torch
from . import Kernel
from gpytorch.lazy import DiagLazyVariable


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
            return None
