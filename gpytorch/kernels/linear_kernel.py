from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from ..lazy import MatmulLazyVariable, RootLazyVariable
from .kernel import Kernel


class LinearKernel(Kernel):
    def __init__(
        self, num_dimensions, variance_bounds=(-10000, 10000), offset_bounds=(-10000, 10000), eps=1e-5, active_dims=None
    ):
        super(LinearKernel, self).__init__(active_dims=active_dims)
        self.eps = eps
        self.register_parameter("variance", nn.Parameter(torch.zeros(1)), bounds=variance_bounds)
        self.register_parameter("offset", nn.Parameter(torch.zeros(1, 1, num_dimensions)), bounds=offset_bounds)

    def forward(self, x1, x2):
        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyVariable when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyVariable(x1 - self.offset)
        else:
            prod = MatmulLazyVariable(x1 - self.offset, (x2 - self.offset).transpose(2, 1))

        return prod + self.variance.expand(prod.size())
