#!/usr/bin/env python3

import torch
from .lazy_tensor import LazyTensor
from ..utils.toeplitz import sym_toeplitz_matmul, sym_toeplitz_derivative_quadratic_form


class ToeplitzLazyTensor(LazyTensor):
    def __init__(self, column):
        super(ToeplitzLazyTensor, self).__init__(column)
        self.column = column

    def _matmul(self, rhs):
        return sym_toeplitz_matmul(self.column, rhs)

    def _t_matmul(self, rhs):
        # Matrix is symmetric
        return self._matmul(rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)

        res = (sym_toeplitz_derivative_quadratic_form(left_vecs, right_vecs),)
        if self.column.ndimension() == 1 and res[0].ndimension() == 2:
            res = (res[0].sum(0),)
        return res

    def _size(self):
        if self.column.ndimension() == 2:
            return torch.Size((self.column.size(0), self.column.size(-1), self.column.size(-1)))
        else:
            return torch.Size((self.column.size(-1), self.column.size(-1)))

    def _transpose_nonbatch(self):
        return ToeplitzLazyTensor(self.column)

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        n_grid = self.column.size(-1)
        toeplitz_indices = (left_indices - right_indices).fmod(n_grid).abs().long()
        return self.column.__getitem__((*batch_indices, toeplitz_indices))

    def add_jitter(self, jitter_val=1e-3):
        jitter = torch.zeros_like(self.column)
        jitter.narrow(-1, 0, 1).fill_(jitter_val)
        return ToeplitzLazyTensor(self.column.add(jitter))

    def diag(self):
        """
        Gets the diagonal of the Toeplitz matrix wrapped by this object.
        """
        diag_term = self.column.select(-1, 0)
        if self.column.ndimension() > 1:
            diag_term = diag_term.unsqueeze(-1)
        return diag_term.expand(*self.column.size())
