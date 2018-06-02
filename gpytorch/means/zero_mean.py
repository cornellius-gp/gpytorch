from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.means import Mean


class ZeroMean(Mean):
    def forward(self, input):
        return torch.zeros_like(input)
