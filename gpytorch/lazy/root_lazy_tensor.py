#!/usr/bin/env python3

import torch

from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify, NonLazyTensor
from .matmul_lazy_tensor import MatmulLazyTensor
from ..utils.broadcasting import _pad_with_singletons
from ..utils.getitem import _noop_index, _equal_indices
from ..utils.memoize import cached


class RootLazyTensor(LazyTensor):
    def __init__(self, root):
        root = lazify(root)
        super(RootLazyTensor, self).__init__(root)
        self.root = root

    def _expand_batch(self, batch_shape):
        return self.__class__(self.root._expand_batch(batch_shape))

    def _get_indices(self, row_index, col_index, *batch_indices):
        row_index = row_index.unsqueeze(-1)
        col_index = col_index.unsqueeze(-1)
        batch_indices = tuple(batch_index.unsqueeze(-1) for batch_index in batch_indices)
        inner_index = torch.arange(0, self.root.size(-1), device=self.device)
        inner_index = _pad_with_singletons(inner_index, row_index.dim() - 1, 0)

        left_tensor = self.root._get_indices(row_index, inner_index, *batch_indices)
        if torch.equal(row_index, col_index):
            res = left_tensor.pow(2).sum(-1)
        else:
            right_tensor = self.root._get_indices(col_index, inner_index, *batch_indices)
            res = (left_tensor * right_tensor).sum(-1)
        return res

    def _getitem(self, row_index, col_index, *batch_indices):
        # Make sure we're not generating more memory with our "efficient" method
        if torch.is_tensor(row_index) and torch.is_tensor(col_index):
            num_indices = row_index.numel()
            if num_indices > self.matrix_shape.numel():
                return lazify(self.evaluate())._getitem(row_index, col_index, *batch_indices)

        left_tensor = self.root._getitem(row_index, _noop_index, *batch_indices)
        if _equal_indices(row_index, col_index):
            res = self.__class__(left_tensor)
        else:
            right_tensor = self.root._getitem(col_index, _noop_index, *batch_indices)
            res = MatmulLazyTensor(left_tensor, right_tensor.transpose(-1, -2))

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

    def _root_decomposition(self):
        return self.root

    def _root_decomposition_size(self):
        return self.root.size(-1)

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
