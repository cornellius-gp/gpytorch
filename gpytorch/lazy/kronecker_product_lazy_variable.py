import torch
import operator
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from functools import reduce


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(sub_matmul_closures, sizes, tensor):
    res = tensor.contiguous()
    is_batch = tensor.ndimension() == 3
    n_cols = tensor.size(-1)
    for sub_matmul_closure, size in list(zip(sub_matmul_closures, sizes))[::-1]:
        if is_batch:
            n_batch = res.size(0)
            res = res.view(n_batch, size, -1)
            factor = sub_matmul_closure(res)
            n_batch = factor.size(0)
            factor = factor.view(n_batch, size, -1, n_cols).transpose(-3, -2)
            res = factor.contiguous().view(n_batch, -1, n_cols)
        else:
            res = res.view(size, -1)
            factor = sub_matmul_closure(res)
            factor = factor.contiguous().view(size, -1, n_cols).transpose(-3, -2)
            res = factor.contiguous().view(-1, n_cols)
    return res


class KroneckerProductLazyVariable(LazyVariable):
    def __init__(self, *lazy_vars):
        if not all(isinstance(lazy_var, LazyVariable) for lazy_var in lazy_vars):
            raise RuntimeError('KroneckerProductLazyVariable is intended to wrap lazy variables.')
        super(KroneckerProductLazyVariable, self).__init__(*lazy_vars)
        self.lazy_vars = lazy_vars

    def _matmul_closure_factory(self, *args):
        sizes = [lazy_var.size(-1) for lazy_var in self.lazy_vars]
        sub_matmul_closures = []
        i = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            sub_matmul_closure = lazy_var._matmul_closure_factory(*args[i:i + len_repr])
            sub_matmul_closures.append(sub_matmul_closure)
            i = i + len_repr

        def closure(tensor):
            is_vec = tensor.ndimension() == 1
            if is_vec:
                tensor = tensor.unsqueeze(-1)

            res = _matmul(sub_matmul_closures, sizes, tensor)
            if is_vec:
                res = res.squeeze(-1)
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        sizes = [lazy_var.size(-1) for lazy_var in self.lazy_vars]
        sub_matmul_closures = []
        sub_derivative_closures = []
        i = 0
        for lazy_var in self.lazy_vars:
            len_repr = len(lazy_var.representation())
            sub_matmul_closure = lazy_var._matmul_closure_factory(*args[i:i + len_repr])
            sub_derivative_closure = lazy_var._derivative_quadratic_form_factory(*args[i:i + len_repr])
            sub_matmul_closures.append(sub_matmul_closure)
            sub_derivative_closures.append(sub_derivative_closure)
            i = i + len_repr

        def closure(left_vectors, right_vectors):
            res = []
            if left_vectors.ndimension() == 1:
                left_vectors = left_vectors.unsqueeze(0)
                right_vectors = right_vectors.unsqueeze(0)

            left_vectors = left_vectors.contiguous()
            right_vectors = right_vectors.contiguous()
            is_batch = left_vectors.ndimension() == 3

            if not is_batch:
                s, m = left_vectors.size()

                for i, sub_derivative_closure in enumerate(sub_derivative_closures):
                    m_left = _prod(sizes[i + 1:])
                    m_right = _prod(sizes[:i])
                    m_i = sizes[i]

                    right_vectors_i = right_vectors.view(s, m_left, m_i, m_right).transpose(2, 3).contiguous()
                    right_vectors_i = right_vectors_i.view(s * m_left * m_right, m_i)

                    left_vectors_i = left_vectors.view(s, int(m / m_left), m_left).transpose(1, 2)
                    if i != len(sub_derivative_closures) - 1:
                        left_vectors_i = left_vectors_i.transpose(0, 1).contiguous().view(m_left, s * int(m / m_left))
                        left_vectors_i = _matmul(sub_matmul_closures[i + 1:], sizes[i + 1:], left_vectors_i)
                        left_vectors_i = left_vectors_i.contiguous().view(m_left, s, int(m / m_left)).transpose(0, 1)

                    left_vectors_i = left_vectors_i.contiguous().view(s, m_left, m_i, m_right).transpose(2, 3)
                    if i != 0:
                        left_vectors_i = left_vectors_i.transpose(0, 2).contiguous().view(m_right, m_left * s * m_i)
                        left_vectors_i = _matmul(sub_matmul_closures[:i], sizes[:i], left_vectors_i)
                        left_vectors_i = left_vectors_i.contiguous().view(m_right, m_left, s, m_i).transpose(0, 2)
                    left_vectors_i = left_vectors_i.contiguous().view(s * m_left * m_right, m_i).contiguous()
                    res = res + list(sub_derivative_closure(left_vectors_i, right_vectors_i))
            else:
                batch_size, s, m = left_vectors.size()

                for i, sub_derivative_closure in enumerate(sub_derivative_closures):
                    m_left = _prod(sizes[i + 1:])
                    m_right = _prod(sizes[:i])
                    m_i = sizes[i]

                    right_vectors_i = right_vectors.view(batch_size, s, m_left, m_i, m_right)
                    right_vectors_i = right_vectors_i.transpose(-2, -1).contiguous()
                    right_vectors_i = right_vectors_i.view(batch_size, s * m_left * m_right, m_i)

                    left_vectors_i = left_vectors.view(batch_size, s, int(m / m_left), m_left).transpose(-2, -1)
                    if i != len(sub_derivative_closures) - 1:
                        left_vectors_i = left_vectors_i.transpose(-2, -3).contiguous()
                        left_vectors_i = left_vectors_i.view(batch_size, m_left, s * int(m / m_left))
                        left_vectors_i = _matmul(sub_matmul_closures[i + 1:], sizes[i + 1:], left_vectors_i)
                        left_vectors_i = left_vectors_i.contiguous().view(batch_size, m_left, s, int(m / m_left))
                        left_vectors_i = left_vectors_i.transpose(-2, -3)

                    left_vectors_i = left_vectors_i.contiguous().view(batch_size, s, m_left, m_i, m_right)
                    left_vectors_i = left_vectors_i.transpose(-2, -1)
                    if i != 0:
                        left_vectors_i = left_vectors_i.transpose(1, 3).contiguous()
                        left_vectors_i = left_vectors_i.view(batch_size, m_right, m_left * s * m_i)
                        left_vectors_i = _matmul(sub_matmul_closures[:i], sizes[:i], left_vectors_i)
                        left_vectors_i = left_vectors_i.contiguous().view(batch_size, m_right, m_left, s, m_i)
                        left_vectors_i = left_vectors_i.transpose(1, 3)
                    left_vectors_i = left_vectors_i.contiguous().view(batch_size, s * m_left * m_right, m_i)
                    left_vectors_i = left_vectors_i.contiguous()
                    res = res + list(sub_derivative_closure(left_vectors_i, right_vectors_i))

            return res
        return closure

    def _size(self):
        left_size = _prod(lazy_var.size(-2) for lazy_var in self.lazy_vars)
        right_size = _prod(lazy_var.size(-1) for lazy_var in self.lazy_vars)

        is_batch = self.lazy_vars[0].ndimension() == 3
        if is_batch:
            return torch.Size((self.lazy_vars[0].size(0), left_size, right_size))
        else:
            return torch.Size((left_size, right_size))

    def _transpose_nonbatch(self):
        return self.__class__(*(lazy_var._transpose_nonbatch() for lazy_var in self.lazy_vars))

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = Variable(self._tensor_cls(left_indices.size()).fill_(1))
        size = self.size(-1)
        for i, lazy_var in enumerate(list(self.lazy_vars)[::-1]):
            size = size / lazy_var.size(-1)
            left_indices_i = left_indices.float().div(size).floor().long()
            right_indices_i = right_indices.float().div(size).floor().long()

            res = res * lazy_var._batch_get_indices(batch_indices, left_indices_i, right_indices_i)
            left_indices = left_indices - (left_indices_i * size)
            right_indices = right_indices - (right_indices_i * size)
        return res

    def _get_indices(self, left_indices, right_indices):
        res = Variable(self._tensor_cls(left_indices.size()).fill_(1))
        size = self.size(-1)
        for i, lazy_var in enumerate(list(self.lazy_vars)[::-1]):
            size = size / lazy_var.size(-1)
            left_indices_i = left_indices.float().div(size).floor().long()
            right_indices_i = right_indices.float().div(size).floor().long()

            res = res * lazy_var._get_indices(left_indices_i, right_indices_i)
            left_indices = left_indices - (left_indices_i * size)
            right_indices = right_indices - (right_indices_i * size)
        return res
