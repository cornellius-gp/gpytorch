import torch
from torch.nn import Module

class Identity(Module):
    def forward(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        else:
            return inputs

