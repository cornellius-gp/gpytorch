#!/usr/bin/env python3

import torch
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor
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

        res = sym_toeplitz_derivative_quadratic_form(left_vecs, right_vecs)

        # Collapse any expanded broadcast dimensions
        if res.dim() > self.column.dim():
            res = res.view(-1, *self.column.shape).sum(0)

        return res,

    def _size(self):
        return torch.Size((*self.column.shape, self.column.size(-1)))

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

    def _getitem(self, *indices):
        n_grid = self.column.size(-1)
        indices = list(indices)
        row_ind = indices[-2]
        col_ind = indices[-1]

        if isinstance(row_ind, int):
            row_ind = torch.tensor([row_ind], device=self.device).long()

        if isinstance(col_ind, int):
            col_ind = torch.tensor([col_ind], device=self.device).long()

        batch_indices = indices[:-2]

        if torch.is_tensor(row_ind) and torch.is_tensor(col_ind):
            # Advanced indexing for Toeplitz matrices is easy
            return self._get_indices(row_ind, col_ind, *batch_indices)
        elif isinstance(row_ind, slice) and torch.is_tensor(col_ind):
            # One slice, one tensor: expand the slice, do 1D advanced indexing
            start = 0 if row_ind.start is None else row_ind.start
            stop = n_grid if row_ind.stop is None else row_ind.stop
            step = 1 if row_ind.step is None else row_ind.step
            row_range = torch.arange(start, stop, step)
            inds = (row_range.unsqueeze(1) - col_ind.unsqueeze(0)).view(-1).fmod(n_grid).abs().long()
            res = self.column.__getitem__((*batch_indices, inds)).view(*batch_indices, row_range.numel(), col_ind.numel())

            if col_ind.numel() == 1:
                res = res.squeeze(-1)

            return res
        elif torch.is_tensor(row_ind) and isinstance(col_ind, slice):
            # One slice, one tensor: expand the slice, do 1D advanced indexing
            start = 0 if col_ind.start is None else col_ind.start
            stop = n_grid if col_ind.stop is None else col_ind.stop
            step = 1 if col_ind.step is None else col_ind.step
            col_range = torch.arange(start, stop, step)
            inds = (row_ind.unsqueeze(1) - col_range.unsqueeze(0)).view(-1).fmod(n_grid).abs().long()
            res = self.column.__getitem__((*batch_indices, inds)).view(*batch_indices, row_ind.numel(), col_range.numel())

            if row_ind.numel() == 1:
                res = res.squeeze(-2)

            return res
        else:
            # Both slices: expand both slices, do 1D advanced indexing
            row_start = 0 if row_ind.start is None else row_ind.start
            row_stop = n_grid if row_ind.stop is None else row_ind.stop
            row_step = 1 if row_ind.step is None else row_ind.step
            col_start = 0 if col_ind.start is None else col_ind.start
            col_stop = n_grid if col_ind.stop is None else col_ind.stop
            col_step = 1 if col_ind.step is None else col_ind.step
            row_range = torch.arange(row_start, row_stop, row_step)
            col_range = torch.arange(col_start, col_stop, col_step)
            inds = (row_range.unsqueeze(1) - col_range.unsqueeze(0)).view(-1).fmod(n_grid).abs().long()
            return NonLazyTensor(self.column.__getitem__((*batch_indices, inds)).view(*batch_indices, row_range.numel(), col_range.numel()))

    def diag(self):
        """
        Gets the diagonal of the Toeplitz matrix wrapped by this object.
        """
        diag_term = self.column[..., 0]
        if self.column.ndimension() > 1:
            diag_term = diag_term.unsqueeze(-1)
        return diag_term.expand(*self.column.size())
