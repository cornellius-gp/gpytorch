from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .mean import Mean


class ZeroMean(Mean):
    def forward(self, input):
        return torch.zeros((input.size(0), input.size(1)), dtype=input.dtype, device=input.device)
