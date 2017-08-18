import torch
from torch.autograd import Function, Variable


class DSMM(Function):
    def __init__(self, sparse):
        if isinstance(sparse, Variable):
            sparse = sparse.data
        self.sparse = sparse

    def forward(self, dense):
        return torch.dsmm(self.sparse, dense)

    def backward(self, grad_output):
        return torch.dsmm(self.sparse.t(), grad_output)
