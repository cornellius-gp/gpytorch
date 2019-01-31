#!/usr/bin/env python3

import torch

from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify, NonLazyTensor
from .matmul_lazy_tensor import MatmulLazyTensor
from ..utils.getitem import _noop_index, _equal_indices
from ..utils.memoize import cached


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class RootLazyTensor(LazyTensor):
    def __init__(self, root):
        root = lazify(root)
        super(RootLazyTensor, self).__init__(root)
        self.root = root

    def _getitem(self, row_col_are_absorbed, row_index, col_index, *batch_indices):
        # Make sure we're not generating more memory with our "efficient" method
        if torch.is_tensor(row_index) and torch.is_tensor(col_index):
            num_indices = row_index.numel()
            if num_indices > self.matrix_shape.numel():
                return lazify(self.evaluate())._getitem(row_col_are_absorbed, row_index, col_index, *batch_indices)

        left_tensor = self.root._getitem(False, row_index, _noop_index, *batch_indices)
        if row_col_are_absorbed and torch.is_tensor(row_index):
            left_tensor = lazify(left_tensor.evaluate().unsqueeze(-2))

        if not _equal_indices(row_index, col_index):
            right_tensor = self.root._getitem(False, col_index, _noop_index, *batch_indices)
            if row_col_are_absorbed and torch.is_tensor(col_index):
                right_tensor = lazify(right_tensor.evaluate().unsqueeze(-2))
            res = MatmulLazyTensor(left_tensor, right_tensor.transpose(-1, -2))
        else:
            res = self.__class__(left_tensor)

        if row_col_are_absorbed:
            res = res.evaluate().squeeze(-2).squeeze(-1)
        return res

    def _matmul(self, rhs):
        return self.root._matmul(self.root._t_matmul(rhs))

    def _t_matmul(self, rhs):
        # Matrix is symmetric
        return self._matmul(rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        right_vecs_times_rhs = self.root._t_matmul(right_vecs)
        left_vecs_times_lhs_t = self.root._t_matmul(left_vecs)

        deriv_part_1 = self.root._quad_form_derivative(left_vecs, right_vecs_times_rhs)
        deriv_part_2 = self.root._quad_form_derivative(right_vecs, left_vecs_times_lhs_t)

        deriv = []
        for item_part_1, item_part_2 in zip(deriv_part_1, deriv_part_2):
            deriv.append(item_part_1 + item_part_2)
        return tuple(deriv)

    def _size(self):
        return torch.Size((*self.root.batch_shape, self.root.size(-2), self.root.size(-2)))

    def _transpose_nonbatch(self):
        return self

    def diag(self):
        if isinstance(self.root, NonLazyTensor):
            return (self.root.tensor ** 2).sum(-1)
        else:
            return super(RootLazyTensor, self).diag()

    @cached
    def evaluate(self):
        eval_root = self.root.evaluate()
        return torch.matmul(eval_root, eval_root.transpose(-1, -2))

    def root_decomposition_size(self):
        return self.root.size(-1)

    def root_decomposition(self):
        return self
