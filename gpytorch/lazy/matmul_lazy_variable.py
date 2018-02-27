import torch
from torch.autograd import Variable
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

    def _matmul_closure_factory(self, *args):
        len_lhs_repr = len(self.lhs.representation())
        lhs_matmul_closure = self.lhs._matmul_closure_factory(*args[:len_lhs_repr])
        rhs_matmul_closure = self.rhs._matmul_closure_factory(*args[len_lhs_repr:])

        def closure(tensor):
            return lhs_matmul_closure(rhs_matmul_closure(tensor))

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        len_lhs_repr = len(self.lhs.representation())
        lhs_t_matmul_closure = self.lhs.transpose(-1, -2)._t_matmul_closure_factory(*args[:len_lhs_repr])
        rhs_matmul_closure = self.rhs._matmul_closure_factory(*args[len_lhs_repr:])
        lhs_derivative_closure = self.lhs._derivative_quadratic_form_factory(*args[:len_lhs_repr])
        rhs_derivative_closure = self.rhs._derivative_quadratic_form_factory(*args[len_lhs_repr:])

        def closure(left_factor, right_factor):
            if left_factor.ndimension() == 1:
                left_factor = left_factor.unsqueeze(0)
                right_factor = right_factor.unsqueeze(0)
            right_factor_times_rhs = rhs_matmul_closure(right_factor.transpose(-1, -2)).transpose(-1, -2)
            left_factor_times_lhs_t = lhs_t_matmul_closure(left_factor.transpose(-1, -2)).transpose(-1, -2)
            left_grad, = lhs_derivative_closure(left_factor, right_factor_times_rhs)
            right_grad, = rhs_derivative_closure(left_factor_times_lhs_t, right_factor)
            return left_grad, right_grad

        return closure

    def _size(self):
        if self.lhs.ndimension() > 2:
            return torch.Size((self.lhs.size(0), self.lhs.size(1), self.rhs.size(2)))
        else:
            return torch.Size((self.lhs.size(0), self.rhs.size(1)))

    def _transpose_nonbatch(self, *args):
        return self.__class__(self.rhs._transpose_nonbatch(), self.lhs._transpose_nonbatch())

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        outer_size = batch_indices.size(0)
        inner_size = self.lhs.size(-1)
        inner_indices = Variable(right_indices.data.new(inner_size))
        torch.arange(0, inner_size, out=inner_indices.data)

        left_vals = self.lhs._batch_get_indices(_outer_repeat(batch_indices, inner_size),
                                                _outer_repeat(left_indices, inner_size),
                                                _inner_repeat(inner_indices, outer_size))
        right_vals = self.rhs._batch_get_indices(_outer_repeat(batch_indices, inner_size),
                                                 _inner_repeat(inner_indices, outer_size),
                                                 _outer_repeat(right_indices, inner_size))

        return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

    def _get_indices(self, left_indices, right_indices):
        outer_size = left_indices.size(0)
        inner_size = self.lhs.size(-1)
        inner_indices = Variable(right_indices.data.new(inner_size))
        torch.arange(0, inner_size, out=inner_indices.data)

        left_vals = self.lhs._get_indices(_outer_repeat(left_indices, inner_size),
                                          _inner_repeat(inner_indices, outer_size))
        right_vals = self.rhs._get_indices(_inner_repeat(inner_indices, outer_size),
                                           _outer_repeat(right_indices, inner_size))

        return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

    def diag(self):
        if isinstance(self.lhs, NonLazyVariable) and isinstance(self.rhs, NonLazyVariable):
            return (self.lhs.var * self.rhs.var.transpose(-1, -2)).sum(-1)
        else:
            return super(MatmulLazyVariable, self).diag()

    def evaluate(self):
        return torch.matmul(self.lhs.evaluate(), self.rhs.evaluate())
