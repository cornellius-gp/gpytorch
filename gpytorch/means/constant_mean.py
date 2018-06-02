from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.means import Mean


class ConstantMean(Mean):
    def __init__(self, constant_bounds=(-1e10, 1e10), batch_size=None):
        super(ConstantMean, self).__init__()
        self.batch_size = batch_size
        if batch_size is None:
            self.register_parameter("constant", torch.nn.Parameter(torch.zeros(1)), bounds=constant_bounds)
        else:
            self.register_parameter("constant", torch.nn.Parameter(torch.zeros(batch_size, 1)), bounds=constant_bounds)

    def forward(self, input):
        if self.batch_size is None:
            return self.constant.expand(input.size(0))
        else:
            return self.constant.expand(input.size(0), input.size(1))
