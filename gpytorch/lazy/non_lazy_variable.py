from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch
from gpytorch.lazy import LazyVariable


class NonLazyVariable(LazyVariable):
    def __init__(self, var):
        """
        Not a lazy variable

        Args:
        - var (Variable: matrix) a variable
        """
        super(NonLazyVariable, self).__init__(var)
        self.tensor = var

    def _matmul(self, rhs):
        return torch.matmul(self.tensor, rhs)

    def _t_matmul(self, rhs):
        return torch.matmul(self.tensor.transpose(-1, -2), rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() < self.tensor.ndimension():
            left_vecs = left_vecs.unsqueeze(-1)
            right_vecs = right_vecs.unsqueeze(-1)

        res = left_vecs.matmul(right_vecs.transpose(-1, -2))
        return res,

    def _size(self):
        return self.tensor.size()

    def _transpose_nonbatch(self):
        return NonLazyVariable(self.tensor.transpose(-1, -2))

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        return self.tensor[batch_indices.data, left_indices.data, right_indices.data]

    def _get_indices(self, left_indices, right_indices):
        return self.tensor[left_indices.data, right_indices.data]

    def add_diag(self, diag):
        return NonLazyVariable(gpytorch.add_diag(self.tensor, diag))

    def diag(self):
        if self.tensor.ndimension() < 3:
            return self.tensor.diag()
        else:
            return super(NonLazyVariable, self).diag()

    def evaluate(self):
        return self.tensor

    def repeat(self, *sizes):
        return NonLazyVariable(self.tensor.repeat(*sizes))

    def __getitem__(self, index):
        return NonLazyVariable(self.tensor[index])
