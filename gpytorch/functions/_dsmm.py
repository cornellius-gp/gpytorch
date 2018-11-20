#!/usr/bin/env python3

import torch
from torch.autograd import Function
from ..utils.sparse import bdsmm


class DSMM(Function):
    def __init__(self, sparse):
        self.sparse = sparse

    def forward(self, dense):
        if self.sparse.ndimension() == 3:
            return bdsmm(self.sparse, dense)
        else:
            return torch.dsmm(self.sparse, dense)

    def backward(self, grad_output):
        if self.sparse.ndimension() == 3:
            return bdsmm(self.sparse.transpose(1, 2), grad_output)
        else:
            return torch.dsmm(self.sparse.t(), grad_output)
