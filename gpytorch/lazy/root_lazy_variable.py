from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class RootLazyVariable(LazyVariable):
    def __init__(self, root):
        if not isinstance(root, LazyVariable):
            root = NonLazyVariable(root)
        super(RootLazyVariable, self).__init__(root)
        self.root = root

    def _matmul(self, rhs):
        return self.root._matmul(self.root._t_matmul(rhs))

    def _t_matmul(self, rhs):
        # Matrix is symmetric
        return self._matmul(rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(0)
            right_vecs = right_vecs.unsqueeze(0)
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

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        outer_size = batch_indices.size(0)
        inner_size = self.root.size(-1)
        inner_indices = Variable(right_indices.data.new(inner_size))
        torch.arange(0, inner_size, out=inner_indices.data)

        left_vals = self.root._batch_get_indices(
            _outer_repeat(batch_indices, inner_size),
            _outer_repeat(left_indices, inner_size),
            _inner_repeat(inner_indices, outer_size),
        )
        right_vals = self.root.transpose(-1, -2)._batch_get_indices(
            _outer_repeat(batch_indices, inner_size),
            _inner_repeat(inner_indices, outer_size),
            _outer_repeat(right_indices, inner_size),
        )

        return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

    def _get_indices(self, left_indices, right_indices):
        outer_size = left_indices.size(0)
        inner_size = self.root.size(-1)
        inner_indices = Variable(right_indices.data.new(inner_size))
        torch.arange(0, inner_size, out=inner_indices.data)

        left_vals = self.root._get_indices(
            _outer_repeat(left_indices, inner_size), _inner_repeat(inner_indices, outer_size)
        )
        right_vals = self.root.t()._get_indices(
            _inner_repeat(inner_indices, outer_size), _outer_repeat(right_indices, inner_size)
        )

        return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

    def diag(self):
        if isinstance(self.root, NonLazyVariable):
            return (self.root.tensor ** 2).sum(-1)
        else:
            return super(RootLazyVariable, self).diag()

    def evaluate(self):
        return torch.matmul(self.root.evaluate(), self.root.transpose(-1, -2).evaluate())

    def root_decomposition_size(self):
        return self.root.size(-1)

    def root_decomposition(self):
        return self.root.evaluate()
