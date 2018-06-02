from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from . import LazyVariable


class ZeroLazyVariable(LazyVariable):
    """
    Special LazyVariable representing zero.
    """

    def __init__(self, *sizes, device):
        super(ZeroLazyVariable, self).__init__(*sizes)
        self.sizes = list(sizes)
        self.device = device

    def _matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-1) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        return rhs * 0

    def _t_matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-1) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        return rhs * 0

    def _quad_form_derivative(self, left_vecs, right_vecs):
        raise RuntimeError("Backwards through a ZeroLazyVariable is not possible")

    def _size(self):
        return torch.Size(self.sizes)

    def add_diag(self, diag):
        from .diag_lazy_variable import DiagLazyVariable

        if diag.size(-1) != self.size(-1):
            raise RuntimeError("Size mismatch, self: {}, diag {}".format(self.size(), diag.size()))

        if self.size(-1) != self.size(-2):
            raise RuntimeError("add_diag only defined for square matrices")

        if self.ndimension() == 3:
            return DiagLazyVariable(diag.unsqueeze(0).expand(self.size(0), self.size(1)))
        else:
            return DiagLazyVariable(diag.expand(self.size(0)))

    def diag(self):
        size = self.size()
        if size[-1] != size[-2]:
            raise RuntimeError("Diag works on square matrices (or batches)")

        if self.ndimension() == 3:
            return torch.zeros(size[-3], size[-2])
        else:
            return torch.zeros(size[-2])

    def evaluate(self):
        return torch.zeros(*self.sizes, device=self.device)

    def inv_matmul(self, tensor):
        raise RuntimeError("ZeroLazyVariables are not invertible!")

    def inv_quad(self, tensor):
        raise RuntimeError("ZeroLazyVariables are not invertible!")

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False):
        raise RuntimeError("ZeroLazyVariables are not invertible!")

    def log_det(self):
        return torch.log(torch.tensor(0.0))

    def matmul(self, tensor):
        tensor_size_ind = -2 if tensor.ndimension() > 1 else -1
        if self.size(-1) != tensor.size(tensor_size_ind):
            raise RuntimeError("Size mismatch, self: {}, tensor: {}".format(self.size(), tensor.size()))
        return tensor * 0

    def mul(self, other):
        return self

    def mul_batch(self, mul_batch_size=None):
        return ZeroLazyVariable(*self.sizes[-2:], self.device)

    def root_decomposition(self):
        raise RuntimeError("ZeroLazyVariables are not positive definite!")

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        raise RuntimeError("ZeroLazyVariables are not positive definite!")

    def root_decomposition_size(self):
        raise RuntimeError("ZeroLazyVariables are not positive definite!")

    def size(self, val=None):
        """
        Returns the size of the resulting Variable that the lazy variable represents
        """
        size = self._size()
        if val is not None:
            return size[val]
        return size

    def sum_batch(self, sum_batch_size=None):
        from .sum_batch_lazy_variable import SumBatchLazyVariable

        return SumBatchLazyVariable(self, sum_batch_size=sum_batch_size)

    def transpose(self, dim1, dim2):
        sizes = self.sizes.copy()
        tmp = sizes[dim1]
        sizes[dim1] = sizes[dim2]
        sizes[dim2] = tmp

        return ZeroLazyVariable(*sizes, device=self.device)

    def __add__(self, other):
        return other

    def __div__(self, other):
        return self

    def __mul__(self, other):
        return self
