import torch
from .lazy_variable import LazyVariable


class MatmulLazyVariable(LazyVariable):
    def __init__(self, lhs, rhs):
        super(MatmulLazyVariable, self).__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs

    def _matmul_closure_factory(self, lhs, rhs):
        def closure(tensor):
            return torch.matmul(lhs, rhs).matmul(tensor)

        return closure

    def _derivative_quadratic_form_factory(self, lhs, rhs):
        def closure(left_factor, right_factor):
            left_grad = left_factor.transpose(-1, -2).matmul(right_factor.matmul(rhs.transpose(-1, -2)))
            right_grad = lhs.transpose(-1, -2).matmul(left_factor.transpose(-1, -2)).matmul(right_factor)
            return left_grad, right_grad

        return closure

    def is_batch(self):
        return self.lhs.ndimension() > 2

    def diag(self):
        return (self.lhs * self.rhs.t()).sum(1)

    def evaluate(self):
        return torch.matmul(self.lhs, self.rhs)

    def size(self):
        if self.is_batch():
            return torch.Size((self.lhs.size()[0], self.lhs.size()[1], self.lhs.size()[1]))
        else:
            return torch.Size((self.lhs.size()[0], self.lhs.size()[0]))

    def __getitem__(self, indices):
        if self.ndimension() == 2:
            index1, index2 = indices
            return MatmulLazyVariable(self.lhs[index1, :], self.rhs[:, index2])
        else:
            raise NotImplementedError('Batch __getitem__ not implemented yet')
