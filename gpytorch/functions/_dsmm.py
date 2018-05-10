from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Function, Variable
from ..utils import bdsmm


class DSMM(Function):

    def __init__(self, sparse):
        if isinstance(sparse, Variable):
            sparse = sparse.data
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
