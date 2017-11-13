import torch
from .lazy_variable import LazyVariable


class MatmulLazyVariable(LazyVariable):
    def __init__(self, lhs, rhs):
        super(MatmulLazyVariable, self).__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs

    def diag(self):
        return (self.lhs * self.rhs.t()).sum(1)

    def evaluate(self):
        return torch.matmul(self.lhs, self.rhs)

    def size(self):
        return torch.Size((self.lhs.size()[0], self.lhs.size()[0]))
