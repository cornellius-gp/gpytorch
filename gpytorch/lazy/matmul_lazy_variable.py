from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class MatmulLazyVariable(LazyVariable):
    def __init__(self, lhs, rhs):
        if not isinstance(lhs, LazyVariable):
            lhs = NonLazyVariable(lhs)
        if not isinstance(rhs, LazyVariable):
            rhs = NonLazyVariable(rhs)

        super(MatmulLazyVariable, self).__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs

    @property
    def _evaluated_lhs(self):
        if not hasattr(self, "_evaluated_lhs_memo"):
            self._evaluated_lhs_memo = self.lhs.evaluate()
        return self._evaluated_lhs_memo

    @property
    def _evaluated_rhs(self):
        if not hasattr(self, "_evaluated_rhs_memo"):
            self._evaluated_rhs_memo = self.rhs.evaluate()
        return self._evaluated_rhs_memo

    def _matmul(self, rhs):
        return self.lhs._matmul(self.rhs._matmul(rhs))

    def _t_matmul(self, rhs):
        return self.rhs._t_matmul(self.lhs._t_matmul(rhs))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        right_vecs_times_rhs = self.rhs._matmul(right_vecs)
        left_vecs_times_lhs_t = self.lhs._t_matmul(left_vecs)
        left_grad, = self.lhs._quad_form_derivative(left_vecs, right_vecs_times_rhs)
        right_grad, = self.rhs._quad_form_derivative(left_vecs_times_lhs_t, right_vecs)
        return left_grad, right_grad

    def _size(self):
        if self.lhs.ndimension() > 2:
            return torch.Size((self.lhs.size(0), self.lhs.size(1), self.rhs.size(2)))
        else:
            return torch.Size((self.lhs.size(0), self.rhs.size(1)))

    def _transpose_nonbatch(self, *args):
        return self.__class__(
            self.rhs._transpose_nonbatch(), self.lhs._transpose_nonbatch()
        )

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        n_indices = left_indices.numel()
        if n_indices > self.size(-1) * self.size(-2) * self.size(-3):
            return self._evaluated[batch_indices, left_indices, right_indices]

        else:
            outer_size = batch_indices.size(0)
            inner_size = self.lhs.size(-1)
            inner_indices = right_indices.new(inner_size)
            torch.arange(0, inner_size, out=inner_indices.data)

            # Repeat the indices to get all the appropriate terms
            batch_indices = _outer_repeat(batch_indices, inner_size)
            left_indices = _outer_repeat(left_indices, inner_size)
            right_indices = _outer_repeat(right_indices, inner_size)
            inner_indices = _inner_repeat(inner_indices, outer_size)

            left_vals = self.lhs._batch_get_indices(
                batch_indices, left_indices, inner_indices
            )
            right_vals = self.rhs._batch_get_indices(
                batch_indices, inner_indices, right_indices
            )

            return (
                left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)
            ).sum(-1)

    def _get_indices(self, left_indices, right_indices):
        n_indices = left_indices.numel()
        if n_indices > self.size(-1) * self.size(-2):
            return self._evaluated[left_indices, right_indices]

        else:
            outer_size = left_indices.size(0)
            inner_size = self.lhs.size(-1)
            inner_indices = right_indices.new(inner_size)
            torch.arange(0, inner_size, out=inner_indices.data)

            # Repeat the indices to get all the appropriate terms
            left_indices = _outer_repeat(left_indices, inner_size)
            right_indices = _outer_repeat(right_indices, inner_size)
            inner_indices = _inner_repeat(inner_indices, outer_size)

            left_vals = self.lhs._get_indices(left_indices, inner_indices)
            right_vals = self.rhs._get_indices(inner_indices, right_indices)

            return (
                left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)
            ).sum(-1)

    def diag(self):
        if isinstance(self.lhs, NonLazyVariable) and isinstance(
            self.rhs, NonLazyVariable
        ):
            return (self.lhs.tensor * self.rhs.tensor.transpose(-1, -2)).sum(-1)
        else:
            return super(MatmulLazyVariable, self).diag()

    def evaluate(self):
        if not hasattr(self, "_evaluated_memo"):
            self._evaluated_memo = torch.matmul(
                self._evaluated_lhs, self._evaluated_rhs
            )
        return self._evaluated_memo
