from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .mean import Mean


class ConstantMean(Mean):
    def __init__(self, prior=None, batch_size=None):
        super(ConstantMean, self).__init__()
        self.batch_size = batch_size
        self.register_parameter(
            name="constant", parameter=torch.nn.Parameter(torch.zeros(batch_size or 1, 1)), prior=prior
        )

    def forward(self, input):
        return self.constant.expand(input.size(0), input.size(1))
