#!/usr/bin/env python3

import operator
from functools import reduce

import torch

from ..utils.broadcasting import _matmul_broadcast_shape
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(lazy_tensors, kp_shape, rhs):
    output_shape = _matmul_broadcast_shape(kp_shape, rhs.shape)
    output_batch_shape = output_shape[:-2]

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for lazy_tensor in lazy_tensors:
        res = res.view(*output_batch_shape, lazy_tensor.size(-1), -1)
        factor = lazy_tensor._matmul(res)
        factor = factor.view(*output_batch_shape, lazy_tensor.size(-2), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


def _t_matmul(lazy_tensors, kp_shape, rhs):
    kp_t_shape = (*kp_shape[:-2], kp_shape[-1], kp_shape[-2])
    output_shape = _matmul_broadcast_shape(kp_t_shape, rhs.shape)
    output_batch_shape = torch.Size(output_shape[:-2])

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for lazy_tensor in lazy_tensors:
        res = res.view(*output_batch_shape, lazy_tensor.size(-2), -1)
        factor = lazy_tensor._t_matmul(res)
        factor = factor.view(*output_batch_shape, lazy_tensor.size(-1), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


class KroneckerProductLazyTensor(LazyTensor):
    def __init__(self, *lazy_tensors):
        try:
            lazy_tensors = tuple(lazify(lazy_tensor) for lazy_tensor in lazy_tensors)
        except TypeError:
            raise RuntimeError("KroneckerProductLazyTensor is intended to wrap lazy tensors.")
        for prev_lazy_tensor, curr_lazy_tensor in zip(lazy_tensors[:-1], lazy_tensors[1:]):
            if prev_lazy_tensor.batch_shape != curr_lazy_tensor.batch_shape:
                raise RuntimeError(
                    "KroneckerProductLazyTensor expects lazy tensors with the "
                    "same batch shapes. Got {}.".format([lv.batch_shape for lv in lazy_tensors])
                )
        super(KroneckerProductLazyTensor, self).__init__(*lazy_tensors)
        self.lazy_tensors = lazy_tensors

    def _get_indices(self, row_index, col_index, *batch_indices):
        row_factor = self.size(-2)
        col_factor = self.size(-1)

        res = None
        for lazy_tensor in self.lazy_tensors:
            sub_row_size = lazy_tensor.size(-2)
            sub_col_size = lazy_tensor.size(-1)

            row_factor //= sub_row_size
            col_factor //= sub_col_size
            sub_res = lazy_tensor._get_indices(
                (row_index // row_factor).fmod(sub_row_size),
                (col_index // col_factor).fmod(sub_col_size),
                *batch_indices,
            )
            res = sub_res if res is None else (sub_res * res)

        return res

    def _matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _matmul(self.lazy_tensors, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    def _t_matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _t_matmul(self.lazy_tensors, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    def _expand_batch(self, batch_shape):
        return self.__class__(*[lazy_tensor._expand_batch(batch_shape) for lazy_tensor in self.lazy_tensors])

    @cached(name="size")
    def _size(self):
        left_size = _prod(lazy_tensor.size(-2) for lazy_tensor in self.lazy_tensors)
        right_size = _prod(lazy_tensor.size(-1) for lazy_tensor in self.lazy_tensors)
        return torch.Size((*self.lazy_tensors[0].batch_shape, left_size, right_size))

    def _transpose_nonbatch(self):
        return self.__class__(*(lazy_tensor._transpose_nonbatch() for lazy_tensor in self.lazy_tensors), **self._kwargs)
