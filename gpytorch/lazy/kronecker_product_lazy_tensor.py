#!/usr/bin/env python3

import torch
import operator
from .lazy_tensor import LazyTensor
from functools import reduce


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(lazy_tensors, rhs):
    res = rhs.contiguous()
    num_cols = rhs.size(-1)
    for lazy_tensor in lazy_tensors:
        res = res.view(*lazy_tensor.batch_shape, lazy_tensor.size(-1), -1)
        factor = lazy_tensor._matmul(res)
        factor = factor.view(*lazy_tensor.batch_shape, lazy_tensor.size(-2), -1, num_cols).transpose(-3, -2)
        res = factor.contiguous().view(*lazy_tensor.batch_shape, -1, num_cols)
    return res


def _t_matmul(lazy_tensors, rhs):
    res = rhs.contiguous()
    num_cols = rhs.size(-1)
    for lazy_tensor in lazy_tensors:
        res = res.view(*lazy_tensor.batch_shape, lazy_tensor.size(-2), -1)
        factor = lazy_tensor._t_matmul(res)
        factor = factor.view(*lazy_tensor.batch_shape, lazy_tensor.size(-1), -1, num_cols).transpose(-3, -2)
        res = factor.contiguous().view(*lazy_tensor.batch_shape, -1, num_cols)
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

    def _size(self):
        left_size = _prod(lazy_tensor.size(-2) for lazy_tensor in self.lazy_tensors)
        right_size = _prod(lazy_tensor.size(-1) for lazy_tensor in self.lazy_tensors)
        return torch.Size((*self.lazy_tensors[0].batch_shape, left_size, right_size))

    def _transpose_nonbatch(self):
        return self.__class__(*(lazy_tensor._transpose_nonbatch() for lazy_tensor in self.lazy_tensors), **self._kwargs)

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        res = torch.ones(left_indices.size(), dtype=self.dtype, device=self.device)
        left_size = self.size(-2)
        right_size = self.size(-1)
        for lazy_tensor in self.lazy_tensors:
            left_size = left_size / lazy_tensor.size(-2)
            right_size = right_size / lazy_tensor.size(-1)
            left_indices_i = left_indices.div(left_size)
            right_indices_i = right_indices.div(right_size)

            res = res * lazy_tensor._get_indices(left_indices_i, right_indices_i, *batch_indices)
            left_indices = left_indices - (left_indices_i * left_size)
            right_indices = right_indices - (right_indices_i * right_size)
        return res
