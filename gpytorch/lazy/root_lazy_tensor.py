#!/usr/bin/env python3

import torch

from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class RootLazyTensor(LazyTensor):
    def __init__(self, root):
        if not isinstance(root, LazyTensor):
            root = NonLazyTensor(root)
        super(RootLazyTensor, self).__init__(root)
        self.root = root

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
        if self.root.ndimension() > 2:
            return torch.Size((self.root.size(0), self.root.size(1), self.root.size(1)))
        else:
            return torch.Size((self.root.size(0), self.root.size(0)))

    def _transpose_nonbatch(self):
        return self

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        n_indices = left_indices.numel()
        if n_indices > self.size(-1) * self.size(-2):
            return self.evaluate().__getitem__((*batch_indices, left_indices, right_indices))

        else:
            outer_size = left_indices.size(0)
            inner_size = self.root.size(-1)
            inner_indices = torch.arange(0, inner_size, dtype=torch.long, device=self.device)

            # Repeat the indices to get all the appropriate terms
            batch_indices = [_outer_repeat(batch_index, inner_size) for batch_index in batch_indices]
            left_indices = _outer_repeat(left_indices, inner_size)
            right_indices = _outer_repeat(right_indices, inner_size)
            inner_indices = _inner_repeat(inner_indices, outer_size)

            left_vals = self.root._get_indices(left_indices, inner_indices, *batch_indices)
            right_vals = self.root.transpose(-1, -2)._get_indices(inner_indices, right_indices, *batch_indices)

        return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

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
