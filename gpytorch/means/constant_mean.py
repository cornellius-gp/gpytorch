import torch
from torch.nn import Parameter
from .mean import Mean

class ConstantMean(Mean):
    def forward(self, input, constant):
        return constant.expand(input.size())
