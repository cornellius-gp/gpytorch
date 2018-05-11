from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.lazy import MatmulLazyVariable, RootLazyVariable
from gpytorch.kernels import Kernel


class LinearKernel(Kernel):
    def __init__(self, num_dimensions, variance_prior=None, offset_prior=None, eps=1e-5, active_dims=None):
        super(LinearKernel, self).__init__(active_dims=active_dims)
        self.eps = eps
        self.register_parameter(name="variance", parameter=torch.nn.Parameter(torch.zeros(1)), prior=variance_prior)
        self.register_parameter(
            name="offset", parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions)), prior=offset_prior
        )

    def forward(self, x1, x2):
        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyVariable when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyVariable(x1 - self.offset)
        else:
            prod = MatmulLazyVariable(x1 - self.offset, (x2 - self.offset).transpose(2, 1))

        return prod + self.variance.expand(prod.size())
