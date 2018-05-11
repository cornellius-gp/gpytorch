from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.means import Mean


class ConstantMean(Mean):
    def __init__(self, batch_size=None, prior=None):
        super(ConstantMean, self).__init__()
        self.batch_size = batch_size
        value = torch.zeros(1) if batch_size is None else torch.zeros(batch_size, 1)
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(value), prior=prior)

    def forward(self, input):
        if self.batch_size is None:
            return self.constant.expand(input.size(0))
        else:
            return self.constant.expand(input.size(0), input.size(1))
