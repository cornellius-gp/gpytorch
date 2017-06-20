import torch
from torch.nn import Module, Parameter

class Bias(Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = Parameter(torch.Tensor(1))
        self.initialize()


    def initialize(self, bias=0):
        self.bias.data.fill_(bias)
        return self


    def forward(self, input):
        return self.bias.expand(input.size())
