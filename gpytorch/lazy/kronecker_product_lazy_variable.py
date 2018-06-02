from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import operator
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from functools import reduce


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(lazy_vars, rhs):
    res = rhs.contiguous()
    is_batch = rhs.ndimension() == 3
    n_cols = rhs.size(-1)
    for lazy_var in list(lazy_vars)[::-1]:
        if is_batch:
            n_batch = res.size(0)
            res = res.view(n_batch, lazy_var.size(-1), -1)
            factor = lazy_var._matmul(res)
            n_batch = factor.size(0)
            factor = factor.view(n_batch, lazy_var.size(-1), -1, n_cols).transpose(-3, -2)
            res = factor.contiguous().view(n_batch, -1, n_cols)
        else:
            res = res.view(lazy_var.size(-1), -1)
            factor = lazy_var._matmul(res)
            factor = factor.contiguous().view(lazy_var.size(-1), -1, n_cols).transpose(-3, -2)
            res = factor.contiguous().view(-1, n_cols)
    return res


class KroneckerProductLazyVariable(LazyVariable):
    def __init__(self, *lazy_vars):
        if not all(isinstance(lazy_var, LazyVariable) for lazy_var in lazy_vars):
            raise RuntimeError("KroneckerProductLazyVariable is intended to wrap lazy variables.")
        super(KroneckerProductLazyVariable, self).__init__(*lazy_vars)
        self.lazy_vars = lazy_vars

    def _matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _matmul(self.lazy_vars, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)

        left_vecs = left_vecs.contiguous()
        right_vecs = right_vecs.contiguous()
        is_batch = left_vecs.ndimension() == 3
        sizes = [lazy_var.size(-1) for lazy_var in self.lazy_vars]

        res = []
        if not is_batch:
            _, s = left_vecs.size()

            for i, lazy_var in enumerate(self.lazy_vars):
                m_left = _prod(sizes[i + 1 :])
                m_right = _prod(sizes[:i])
                m_i = sizes[i]

                right_vecs_i = right_vecs.view(m_left, m_i, m_right, s).transpose(0, 1).contiguous()
                right_vecs_i = right_vecs_i.view(m_i, m_left * m_right * s)

                left_vecs_i = left_vecs.view(m_left, m_i * m_right, s)
                if i != len(self.lazy_vars) - 1:
                    left_vecs_i = left_vecs_i.view(m_left, m_i * m_right * s)
                    left_vecs_i = _matmul(self.lazy_vars[i + 1 :], left_vecs_i)
                    left_vecs_i = left_vecs_i.view(m_left, m_i * m_right, s)

                left_vecs_i = left_vecs_i.view(m_left, m_i, m_right, s).transpose(0, 1)
                if i != 0:
                    left_vecs_i = left_vecs_i.transpose(0, 2).contiguous().view(m_right, m_left * m_i * s)
                    left_vecs_i = _matmul(self.lazy_vars[:i], left_vecs_i)
                    left_vecs_i = left_vecs_i.view(m_right, m_left, m_i, s).transpose(0, 2)

                left_vecs_i = left_vecs_i.contiguous().view(m_i, m_left * m_right * s).contiguous()
                res = res + list(lazy_var._quad_form_derivative(left_vecs_i, right_vecs_i))
        else:
            batch_size, _, s = left_vecs.size()

            for i, lazy_var in enumerate(self.lazy_vars):
                m_left = _prod(sizes[i + 1 :])
                m_right = _prod(sizes[:i])
                m_i = sizes[i]

                right_vecs_i = right_vecs.view(batch_size, m_left, m_i, m_right, s)
                right_vecs_i = right_vecs_i.transpose(1, 2).contiguous()
                right_vecs_i = right_vecs_i.view(batch_size, m_i, m_left * m_right * s)

                left_vecs_i = left_vecs.view(batch_size, m_left, m_i * m_right, s)
                if i != len(self.lazy_vars) - 1:
                    left_vecs_i = left_vecs_i.view(batch_size, m_left, m_i * m_right * s)
                    left_vecs_i = _matmul(self.lazy_vars[i + 1 :], left_vecs_i)
                    left_vecs_i = left_vecs_i.contiguous().view(batch_size, m_left, m_i * m_right, s)

                left_vecs_i = left_vecs_i.view(batch_size, m_left, m_i, m_right, s)
                left_vecs_i = left_vecs_i.transpose(1, 2)
                if i != 0:
                    left_vecs_i = left_vecs_i.transpose(1, 3).contiguous()
                    left_vecs_i = left_vecs_i.view(batch_size, m_right, m_left * m_i * s)
                    left_vecs_i = _matmul(self.lazy_vars[:i], left_vecs_i)
                    left_vecs_i = left_vecs_i.view(batch_size, m_right, m_left, m_i, s)
                    left_vecs_i = left_vecs_i.transpose(1, 3)
                left_vecs_i = left_vecs_i.contiguous().view(batch_size, m_i, m_left * m_right * s)
                left_vecs_i = left_vecs_i.contiguous()
                res = res + list(lazy_var._quad_form_derivative(left_vecs_i, right_vecs_i))

        return res

    def _size(self):
        left_size = _prod(lazy_var.size(-2) for lazy_var in self.lazy_vars)
        right_size = _prod(lazy_var.size(-1) for lazy_var in self.lazy_vars)

        is_batch = self.lazy_vars[0].ndimension() == 3
        if is_batch:
            return torch.Size((self.lazy_vars[0].size(0), left_size, right_size))
        else:
            return torch.Size((left_size, right_size))

    def _transpose_nonbatch(self):
        return self.__class__(*(lazy_var._transpose_nonbatch() for lazy_var in self.lazy_vars), **self._kwargs)

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = Variable(self.tensor_cls(left_indices.size()).fill_(1))
        size = self.size(-1)
        for lazy_var in list(self.lazy_vars)[::-1]:
            size = size / lazy_var.size(-1)
            left_indices_i = left_indices.div(size)
            right_indices_i = right_indices.div(size)

            res = res * lazy_var._batch_get_indices(batch_indices, left_indices_i, right_indices_i)
            left_indices = left_indices - (left_indices_i * size)
            right_indices = right_indices - (right_indices_i * size)
        return res

    def _get_indices(self, left_indices, right_indices):
        res = Variable(self.tensor_cls(left_indices.size()).fill_(1))
        size = self.size(-1)
        for lazy_var in list(self.lazy_vars)[::-1]:
            size = size / lazy_var.size(-1)
            left_indices_i = left_indices.div(size)
            right_indices_i = right_indices.div(size)

            res = res * lazy_var._get_indices(left_indices_i, right_indices_i)
            left_indices = left_indices - (left_indices_i * size)
            right_indices = right_indices - (right_indices_i * size)
        return res

    def repeat(self, *sizes):
        return KroneckerProductLazyVariable(*[lazy_var.repeat(*sizes) for lazy_var in self.lazy_vars])
