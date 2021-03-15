#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _pad_with_singletons
from ..utils.getitem import _equal_indices, _noop_index
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .matmul_lazy_tensor import MatmulLazyTensor
from .non_lazy_tensor import NonLazyTensor, lazify


class RootLazyTensor(LazyTensor):
    def __init__(self, root):
        root = lazify(root)
        super().__init__(root)
        self.root = root

    def _expand_batch(self, batch_shape):
        if len(batch_shape) == 0:
            return self
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

    def _mul_constant(self, constant):
        if constant > 0:
            res = self.__class__(self.root._mul_constant(constant.sqrt()))
        else:
            res = super()._mul_constant(constant)
        return res

    def _t_matmul(self, rhs):
        # Matrix is symmetric
        return self._matmul(rhs)

    def add_low_rank(self, low_rank_mat, root_decomp_method=None, root_inv_decomp_method="pinverse"):
        return super().add_low_rank(low_rank_mat, root_inv_decomp_method=root_inv_decomp_method)

    def root_decomposition(self, method=None):
        return self

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
            return super().diag()

    @cached
    def evaluate(self):
        eval_root = self.root.evaluate()
        return torch.matmul(eval_root, eval_root.transpose(-1, -2))
