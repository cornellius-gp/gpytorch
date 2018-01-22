import torch
from .lazy_variable import LazyVariable


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class MatmulLazyVariable(LazyVariable):
    def __init__(self, lhs, rhs):
        super(MatmulLazyVariable, self).__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs

    def _matmul_closure_factory(self, lhs, rhs):
        def closure(tensor):
            return torch.matmul(lhs, rhs).matmul(tensor)

        return closure

    def _derivative_quadratic_form_factory(self, lhs, rhs):
        def closure(left_factor, right_factor):
            if left_factor.ndimension() == 1:
                left_factor = left_factor.unsqueeze(0)
                right_factor = right_factor.unsqueeze(0)
            left_grad = left_factor.transpose(-1, -2).matmul(right_factor.matmul(rhs.transpose(-1, -2)))
            right_grad = lhs.transpose(-1, -2).matmul(left_factor.transpose(-1, -2)).matmul(right_factor)
            return left_grad, right_grad

        return closure

    def _t_matmul_closure_factory(self, *args):
        len_lhs_repr = len(self.lhs.representation())
        lhs_matmul_closure = self.lhs._t_matmul_closure_factory(*args[:len_lhs_repr])
        rhs_matmul_closure = self.rhs._t_matmul_closure_factory(*args[len_lhs_repr:])

        def closure(tensor):
            return rhs_matmul_closure(lhs_matmul_closure(tensor))

        return closure

    def _size(self):
        if self.lhs.ndimension() > 2:
            return torch.Size((self.lhs.size()[0], self.lhs.size()[1], self.lhs.size()[1]))
        else:
            return torch.Size((self.lhs.size()[0], self.lhs.size()[0]))

    def _transpose_nonbatch(self):
        return MatmulLazyVariable(self.rhs.transpose(-1, -2), self.lhs.transpose(-1, -2))

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        outer_size = batch_indices.size(0)
        batch_indices = batch_indices.data
        left_indices = left_indices.data
        right_indices = right_indices.data

        inner_size = self.lhs.size(-1)
        inner_indices = right_indices.new(inner_size)
        torch.arange(0, inner_size, out=inner_indices)

        left_vals = self.lhs[_outer_repeat(batch_indices, inner_size), _outer_repeat(left_indices, inner_size),
                             _inner_repeat(inner_indices, outer_size)]
        right_vals = self.rhs[_outer_repeat(batch_indices, inner_size), _inner_repeat(inner_indices, outer_size),
                              _outer_repeat(right_indices, inner_size)]
        return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

    def _get_indices(self, left_indices, right_indices):
        res = self.lhs.index_select(-2, left_indices) * self.rhs.index_select(-1, right_indices).transpose(-1, -2)
        return res.sum(-1)

    def diag(self):
        return (self.lhs * self.rhs.transpose(-1, -2)).sum(-1)

    def evaluate(self):
        return torch.matmul(self.lhs, self.rhs)
