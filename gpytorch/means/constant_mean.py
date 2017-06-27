import torch
from torch.nn import Parameter
from .mean import Mean

class ConstantMean(Mean):
    def __init__(self):
        super(ConstantMean, self).__init__()
        self.constant = Parameter(torch.Tensor(1))
        self.initialize()


    def initialize(self, constant=0):
        self.constant.data.fill_(constant)
        return self


    def forward(self, input):
        return self.constant.expand(input.size())
