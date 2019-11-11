#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _pad_with_singletons
from ..utils.getitem import _noop_index
from .block_lazy_tensor import BlockLazyTensor


class SumBatchLazyTensor(BlockLazyTensor):
    """
    Represents a lazy tensor that is actually the sum of several lazy tensors blocks.
    The :attr:`block_dim` attribute specifies which dimension of the base LazyTensor
    specifies the blocks.
    For example, (with `block_dim=-3` a `k x n x n` tensor represents `k` `n x n` blocks (a `n x n` matrix).
    A `b x k x n x n` tensor represents `k` `b x n x n` blocks (a `b x n x n` batch matrix).

    Args:
        :attr:`base_lazy_tensor` (LazyTensor):
            A `k x n x n` LazyTensor, or a `b x k x n x n` LazyTensor.
        :attr:`block_dim` (int):
            The dimension that specifies the blocks.
    """

    def _add_batch_dim(self, other):
        shape = list(other.shape)
        expand_shape = list(other.shape)
        shape.insert(-2, 1)
        expand_shape.insert(-2, self.base_lazy_tensor.size(-3))
        other = other.reshape(*shape).expand(*expand_shape)
        return other

    def _get_indices(self, row_index, col_index, *batch_indices):
        # Create an extra index for the summed dimension
        sum_index = torch.arange(0, self.base_lazy_tensor.size(-3), device=self.device)
        sum_index = _pad_with_singletons(sum_index, row_index.dim(), 0)
        row_index = row_index.unsqueeze(-1)
        col_index = col_index.unsqueeze(-1)
        batch_indices = [index.unsqueeze(-1) for index in batch_indices]

        res = self.base_lazy_tensor._get_indices(row_index, col_index, *batch_indices, sum_index)
        return res.sum(-1)

    def _getitem(self, row_index, col_index, *batch_indices):
        res = self.base_lazy_tensor._getitem(row_index, col_index, *batch_indices, _noop_index)
        return self.__class__(res, **self._kwargs)

    def _remove_batch_dim(self, other):
        return other.sum(-3)

    def _size(self):
        shape = list(self.base_lazy_tensor.shape)
        del shape[-3]
        return torch.Size(shape)

    def diag(self):
        diag = self.base_lazy_tensor.diag().sum(-2)
        return diag

    def evaluate(self):
        return self.base_lazy_tensor.evaluate().sum(dim=-3)  # BlockLazyTensors always use dim3 for the block_dim
