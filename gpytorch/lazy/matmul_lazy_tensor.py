#!/usr/bin/env python3

import torch

from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify, NonLazyTensor
from ..utils.getitem import _noop_index
from ..utils.memoize import cached


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class MatmulLazyTensor(LazyTensor):
    def __init__(self, left_lazy_tensor, right_lazy_tensor):
        left_lazy_tensor = lazify(left_lazy_tensor)
        right_lazy_tensor = lazify(right_lazy_tensor)

        super(MatmulLazyTensor, self).__init__(left_lazy_tensor, right_lazy_tensor)
        self.left_lazy_tensor = left_lazy_tensor
        self.right_lazy_tensor = right_lazy_tensor

    def _getitem(self, row_col_are_absorbed, row_index, col_index, *batch_indices):
        # If only some of the batch indices are tensor indexed, then some weird behavior occurs
        # We'll just default to the standard behavior in this case
        batch_indices_are_tensors = [torch.is_tensor(batch_index) for batch_index in batch_indices]
        if any(batch_indices_are_tensors) and not all(batch_indices_are_tensors):
            return super(MatmulLazyTensor, self)._getitem(row_col_are_absorbed, row_index, col_index, *batch_indices)

        # Make sure we're not generating more memory with our "efficient" method
        if torch.is_tensor(row_index) and torch.is_tensor(col_index):
            num_indices = row_index.numel()
            if num_indices > self.matrix_shape.numel():
                return lazify(self.evaluate())._getitem(row_col_are_absorbed, row_index, col_index, *batch_indices)

        left_tensor = self.left_lazy_tensor._getitem(False, row_index, _noop_index, *batch_indices)
        if row_col_are_absorbed and torch.is_tensor(row_index):
            left_tensor = lazify(left_tensor.evaluate().unsqueeze(-2))
        right_tensor = self.right_lazy_tensor._getitem(False, _noop_index, col_index, *batch_indices)
        if row_col_are_absorbed and torch.is_tensor(col_index):
            right_tensor = lazify(right_tensor.evaluate().unsqueeze(-1))

        res = MatmulLazyTensor(left_tensor, right_tensor)
        if row_col_are_absorbed:
            res = res.evaluate().squeeze(-2).squeeze(-1)
        return res

    def _matmul(self, right_lazy_tensor):
        return self.left_lazy_tensor._matmul(self.right_lazy_tensor._matmul(right_lazy_tensor))

    def _t_matmul(self, right_lazy_tensor):
        return self.right_lazy_tensor._t_matmul(self.left_lazy_tensor._t_matmul(right_lazy_tensor))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        right_vecs_times_right_lazy_tensor = self.right_lazy_tensor._matmul(right_vecs)
        left_vecs_times_left_lazy_tensor_t = self.left_lazy_tensor._t_matmul(left_vecs)
        left_grad = self.left_lazy_tensor._quad_form_derivative(left_vecs, right_vecs_times_right_lazy_tensor)
        right_grad = self.right_lazy_tensor._quad_form_derivative(left_vecs_times_left_lazy_tensor_t, right_vecs)

        left_grad = (left_grad,) if not isinstance(left_grad, tuple) else left_grad
        right_grad = (right_grad,) if not isinstance(right_grad, tuple) else right_grad
        return left_grad + right_grad

    def _size(self):
        return torch.Size(
            (*self.left_lazy_tensor.batch_shape, self.left_lazy_tensor.size(-2), self.right_lazy_tensor.size(-1))
        )

    def _transpose_nonbatch(self, *args):
        return self.__class__(self.right_lazy_tensor._transpose_nonbatch(), self.left_lazy_tensor._transpose_nonbatch())

    def diag(self):
        if isinstance(self.left_lazy_tensor, NonLazyTensor) and isinstance(self.right_lazy_tensor, NonLazyTensor):
            return (self.left_lazy_tensor.tensor * self.right_lazy_tensor.tensor.transpose(-1, -2)).sum(-1)
        else:
            return super(MatmulLazyTensor, self).diag()

    @cached
    def evaluate(self):
        return torch.matmul(self.left_lazy_tensor.evaluate(), self.right_lazy_tensor.evaluate())
