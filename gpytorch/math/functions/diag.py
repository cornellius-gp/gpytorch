import torch
from torch.autograd import Function

class Diag(Function):
    def __init__(self, size):
        self.size = size

    
    def forward(self, input):
        assert(input.numel() == 1, 'Input must be a single-element tensor')
        val = input.squeeze()[0]
        res = torch.eye(self.size).type_as(input)
        res.mul_(val)
        return res


    def backward(self, grad_output):
        res = grad_output.new().resize_(1)
        res.fill_(grad_output.trace())
        return res
