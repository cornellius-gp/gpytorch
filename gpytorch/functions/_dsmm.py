#!/usr/bin/env python3

from torch.autograd import Function
from ..utils.sparse import bdsmm


class DSMM(Function):
    def __init__(self, sparse):
        self.sparse = sparse

    def forward(self, dense):
        return bdsmm(self.sparse, dense)

    def backward(self, grad_output):
        return bdsmm(self.sparse.transpose(-1, -2), grad_output)
