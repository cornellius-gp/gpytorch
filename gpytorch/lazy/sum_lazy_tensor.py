#!/usr/bin/env python3
from torch import Tensor

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify
from .zero_lazy_tensor import ZeroLazyTensor

# from .broadcasted_lazy_tensor import BroadcastedLazyTensor


class SumLazyTensor(LazyTensor):
    def __init__(self, *lazy_tensors, **kwargs):
        try:
            lazy_tensors = tuple(lazify(lt) for lt in lazy_tensors)
        except TypeError:
            raise TypeError("All arguments of a SumLazyTensor should be LazyTensors or Tensors")
        # batch_shape = _mul_broadcast_shape(*[lt.shape for lt in lazy_tensors])
        # lazy_tensors = tuple(lt.expand(batch_shape) for lt in lazy_tensors)
        super(SumLazyTensor, self).__init__(*lazy_tensors, **kwargs)

        self.lazy_tensors = lazy_tensors

    def _expand_batch(self, batch_shape):
        expanded_tensors = [lazy_tensor._expand_batch(batch_shape) for lazy_tensor in self.lazy_tensors]
        return self.__class__(*expanded_tensors)

    def _get_indices(self, row_index, col_index, *batch_indices):
        results = [lazy_tensor._get_indices(row_index, col_index, *batch_indices) for lazy_tensor in self.lazy_tensors]
        return sum(results)

    def _getitem(self, row_index, col_index, *batch_indices):
        results = [lazy_tensor._getitem(row_index, col_index, *batch_indices) for lazy_tensor in self.lazy_tensors]
        return SumLazyTensor(*results)

    def _matmul(self, rhs):
        return sum(lazy_tensor._matmul(rhs) for lazy_tensor in self.lazy_tensors)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return tuple(
            var for lazy_tensor in self.lazy_tensors for var in lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        )

    def _size(self):
        return _mul_broadcast_shape(*[lt.shape for lt in self.lazy_tensors])

    def _sum_batch(self, dim):
        return self.__class__(*(lazy_tensor._sum_batch(dim) for lazy_tensor in self.lazy_tensors))

    def _t_matmul(self, rhs):
        return sum(lazy_tensor._t_matmul(rhs) for lazy_tensor in self.lazy_tensors)

    def _transpose_nonbatch(self):
        lazy_tensors_t = [lazy_tensor.transpose(-1, -2) for lazy_tensor in self.lazy_tensors]
        return self.__class__(*lazy_tensors_t)

    @cached
    def evaluate(self):
        return sum(lazy_tensor.evaluate() for lazy_tensor in self.lazy_tensors)

    def __add__(self, other):
        from .diag_lazy_tensor import DiagLazyTensor
        from .added_diag_lazy_tensor import AddedDiagLazyTensor

        if isinstance(other, ZeroLazyTensor):
            return self
        elif isinstance(other, DiagLazyTensor):
            return AddedDiagLazyTensor(self, other)
        elif isinstance(other, SumLazyTensor):
            return SumLazyTensor(*(list(self.lazy_tensors) + list(other.lazy_tensors)))
        elif isinstance(other, LazyTensor):
            return SumLazyTensor(*(list(self.lazy_tensors) + [other]))
        elif isinstance(other, Tensor):
            # get broadcast shape, assuming mul broadcasting the same as add broadcasting
            broadcasted_shape = _mul_broadcast_shape(self.shape, other.shape)

            # lazify + broadcast other
            broadcasted_other = lazify(other.expand(broadcasted_shape))

            # update the lazy tensors' shape as well
            if broadcasted_shape != self.shape:
                broadcasted_lts = [
                    lt.expand(*broadcasted_shape, 1).squeeze(-1).transpose(-1, -2) for lt in self.lazy_tensors
                ]
            else:
                broadcasted_lts = list(self.lazy_tensors)

            return SumLazyTensor(*(broadcasted_lts + [broadcasted_other]))
        else:
            raise AttributeError("other must be a LazyTensor")

    def diag(self):
        return sum(lazy_tensor.diag().contiguous() for lazy_tensor in self.lazy_tensors)
