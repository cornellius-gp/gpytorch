from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import operator
from .lazy_tensor import LazyTensor
from functools import reduce


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(lazy_tensors, rhs):
    res = rhs.contiguous()
    is_batch = rhs.ndimension() == 3
    num_cols = rhs.size(-1)
    for lazy_tensor in list(lazy_tensors)[::-1]:
        if is_batch:
            num_batch = res.size(0)
            res = res.view(num_batch, lazy_tensor.size(-1), -1)
            factor = lazy_tensor._matmul(res)
            factor = factor.view(num_batch, lazy_tensor.size(-2), -1, num_cols)
            factor = factor.transpose(-3, -2).contiguous().view(-1, num_cols)
            res = factor.contiguous().view(num_batch, -1, num_cols)
        else:
            res = res.view(lazy_tensor.size(-1), -1)
            factor = lazy_tensor._matmul(res)
            factor = factor.view(lazy_tensor.size(-2), -1, num_cols).transpose(-3, -2).contiguous().view(-1, num_cols)
            res = factor.contiguous().view(-1, num_cols)
    return res


def _t_matmul(lazy_tensors, rhs):
    res = rhs.contiguous()
    is_batch = rhs.ndimension() == 3
    num_cols = rhs.size(-1)
    for lazy_tensor in list(lazy_tensors)[::-1]:
        if is_batch:
            num_batch = res.size(0)
            res = res.view(num_batch, lazy_tensor.size(-2), -1)
            factor = lazy_tensor._t_matmul(res)
            factor = factor.view(num_batch, lazy_tensor.size(-1), -1, num_cols)
            factor = factor.transpose(-3, -2).contiguous().view(-1, num_cols)
            res = factor.contiguous().view(num_batch, -1, num_cols)
        else:
            res = res.view(lazy_tensor.size(-2), -1)
            factor = lazy_tensor._t_matmul(res)
            factor = factor.view(lazy_tensor.size(-1), -1, num_cols).transpose(-3, -2).contiguous().view(-1, num_cols)
            res = factor.contiguous().view(-1, num_cols)
    return res


class KroneckerProductLazyTensor(LazyTensor):
    def __init__(self, *lazy_tensors):
        if not all(isinstance(lazy_tensor, LazyTensor) for lazy_tensor in lazy_tensors):
            raise RuntimeError("KroneckerProductLazyTensor is intended to wrap lazy tensors.")
        for prev_lazy_tensor, curr_lazy_tensor in zip(lazy_tensors[:-1], lazy_tensors[1:]):
            if prev_lazy_tensor.ndimension() != curr_lazy_tensor.ndimension():
                raise RuntimeError(
                    "KroneckerProductLazyTensor expects lazy tensors with the "
                    "same number of dimensions. Got {}.".format([lv.ndimension() for lv in lazy_tensors])
                )
            if curr_lazy_tensor.ndimension() >= 3:
                if prev_lazy_tensor.shape[:-2] != curr_lazy_tensor.shape[:-2]:
                    raise RuntimeError(
                        "KroneckerProductLazyTensor expects the same batch sizes for component tensors. "
                        "Got sizes: {}".format([lv.size() for lv in lazy_tensors])
                    )
        super(KroneckerProductLazyTensor, self).__init__(*lazy_tensors)
        self.lazy_tensors = lazy_tensors

    def _matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _matmul(self.lazy_tensors, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    def _t_matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _t_matmul(self.lazy_tensors, rhs.contiguous())

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
        row_sizes = [lazy_tensor.size(-2) for lazy_tensor in self.lazy_tensors]
        col_sizes = [lazy_tensor.size(-1) for lazy_tensor in self.lazy_tensors]

        res = []
        if not is_batch:
            _, s = left_vecs.size()

            # We need to call the derivative routine for each of the sub matrices
            for i, lazy_tensor in enumerate(self.lazy_tensors):
                m_left_prev = _prod(row_sizes[i + 1 :])
                m_left_next = _prod(row_sizes[:i])
                m_left_i = row_sizes[i]
                m_right_prev = _prod(col_sizes[i + 1 :])
                m_right_next = _prod(col_sizes[:i])
                m_right_i = col_sizes[i]

                # The right vectors are the same for each of the sub matrices
                # Here we're reshaping them so that the batch dimension corresponds to the correct vectors
                right_vecs_i = right_vecs.view(m_right_prev, m_right_i, m_right_next, s).transpose(0, 1).contiguous()
                right_vecs_i = right_vecs_i.view(m_right_i, m_right_prev * m_right_next * s)

                # The left vectors need to be multiplied by all other sub matrices in the kronecker product
                # First multiply by all the matrices in the KP that are to follow this matrix
                if i != len(self.lazy_tensors) - 1:
                    left_vecs_i = left_vecs.view(m_left_prev, m_left_i * m_left_next * s)
                    left_vecs_i = _t_matmul(self.lazy_tensors[i + 1 :], left_vecs_i)
                    m_left_prev = m_right_prev
                    left_vecs_i = left_vecs_i.view(m_left_prev, m_left_i * m_left_next, s)
                else:
                    left_vecs_i = left_vecs.view(m_left_prev, m_left_i * m_left_next, s)

                # Next multiply by all the previous matrices in the KP
                left_vecs_i = left_vecs_i.view(m_left_prev, m_left_i, m_left_next, s).transpose(0, 1)
                if i != 0:
                    left_vecs_i = left_vecs_i.transpose(0, 2).contiguous().view(m_left_next, m_left_prev * m_left_i * s)
                    left_vecs_i = _t_matmul(self.lazy_tensors[:i], left_vecs_i)
                    m_left_next = m_right_next
                    left_vecs_i = left_vecs_i.view(m_left_next, m_left_prev, m_left_i, s).transpose(0, 2)

                # Now reshape the left vectors
                left_vecs_i = left_vecs_i.contiguous().view(m_left_i, m_left_prev * m_left_next * s).contiguous()
                res = res + list(lazy_tensor._quad_form_derivative(left_vecs_i, right_vecs_i))
        else:
            batch_size, _, s = left_vecs.size()

            for i, lazy_tensor in enumerate(self.lazy_tensors):
                m_right_prev = _prod(col_sizes[i + 1 :])
                m_right_next = _prod(col_sizes[:i])
                m_right_i = col_sizes[i]
                m_left_prev = _prod(row_sizes[i + 1 :])
                m_left_next = _prod(row_sizes[:i])
                m_left_i = row_sizes[i]

                right_vecs_i = right_vecs.view(batch_size, m_right_prev, m_right_i, m_right_next, s)
                right_vecs_i = right_vecs_i.transpose(1, 2).contiguous()
                right_vecs_i = right_vecs_i.view(batch_size, m_right_i, m_right_prev * m_right_next * s)

                if i != len(self.lazy_tensors) - 1:
                    left_vecs_i = left_vecs.view(batch_size, m_left_prev, m_left_i * m_left_next * s)
                    left_vecs_i = _t_matmul(self.lazy_tensors[i + 1 :], left_vecs_i)
                    m_left_prev = m_right_prev
                    left_vecs_i = left_vecs_i.contiguous().view(batch_size, m_left_prev, m_left_i * m_left_next, s)
                else:
                    left_vecs_i = left_vecs.view(batch_size, m_left_prev, m_left_i * m_left_next, s)

                left_vecs_i = left_vecs_i.view(batch_size, m_left_prev, m_left_i, m_left_next, s)
                left_vecs_i = left_vecs_i.transpose(1, 2)
                if i != 0:
                    left_vecs_i = left_vecs_i.transpose(1, 3).contiguous()
                    left_vecs_i = left_vecs_i.view(batch_size, m_left_next, m_left_prev * m_left_i * s)
                    left_vecs_i = _t_matmul(self.lazy_tensors[:i], left_vecs_i)
                    m_left_next = m_right_next
                    left_vecs_i = left_vecs_i.view(batch_size, m_left_next, m_left_prev, m_left_i, s)
                    left_vecs_i = left_vecs_i.transpose(1, 3)
                left_vecs_i = left_vecs_i.contiguous().view(batch_size, m_left_i, m_left_prev * m_left_next * s)
                left_vecs_i = left_vecs_i.contiguous()
                res = res + list(lazy_tensor._quad_form_derivative(left_vecs_i, right_vecs_i))

        return res

    def _size(self):
        left_size = _prod(lazy_tensor.size(-2) for lazy_tensor in self.lazy_tensors)
        right_size = _prod(lazy_tensor.size(-1) for lazy_tensor in self.lazy_tensors)

        is_batch = self.lazy_tensors[0].ndimension() == 3
        if is_batch:
            return torch.Size((self.lazy_tensors[0].size(0), left_size, right_size))
        else:
            return torch.Size((left_size, right_size))

    def _transpose_nonbatch(self):
        return self.__class__(*(lazy_tensor._transpose_nonbatch() for lazy_tensor in self.lazy_tensors), **self._kwargs)

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = torch.ones(left_indices.size(), dtype=self.dtype, device=self.device)
        left_size = self.size(-2)
        right_size = self.size(-1)
        for lazy_tensor in list(self.lazy_tensors)[::-1]:
            left_size = left_size / lazy_tensor.size(-2)
            right_size = right_size / lazy_tensor.size(-1)
            left_indices_i = left_indices.div(left_size)
            right_indices_i = right_indices.div(right_size)

            res = res * lazy_tensor._batch_get_indices(batch_indices, left_indices_i, right_indices_i)
            left_indices = left_indices - (left_indices_i * left_size)
            right_indices = right_indices - (right_indices_i * right_size)
        return res

    def _get_indices(self, left_indices, right_indices):
        res = torch.ones(left_indices.size(), dtype=self.dtype, device=self.device)
        left_size = self.size(-2)
        right_size = self.size(-1)
        for lazy_tensor in list(self.lazy_tensors)[::-1]:
            left_size = left_size / lazy_tensor.size(-2)
            right_size = right_size / lazy_tensor.size(-1)
            left_indices_i = left_indices.div(left_size)
            right_indices_i = right_indices.div(right_size)

            res = res * lazy_tensor._get_indices(left_indices_i, right_indices_i)
            left_indices = left_indices - (left_indices_i * left_size)
            right_indices = right_indices - (right_indices_i * right_size)
        return res

    def repeat(self, *sizes):
        return KroneckerProductLazyTensor(*[lazy_tensor.repeat(*sizes) for lazy_tensor in self.lazy_tensors])
