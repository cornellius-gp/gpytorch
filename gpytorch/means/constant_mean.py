import torch
from torch import nn
from .mean import Mean


class ConstantMean(Mean):
    def __init__(self, constant_bounds=(-1e10, 1e10)):
        super(ConstantMean, self).__init__()
        self.register_parameter('constant', nn.Parameter(torch.zeros(1)), bounds=constant_bounds)

    def forward(self, input):
        return self.constant.expand(input.size()[0])
