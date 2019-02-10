#!/usr/bin/env python3

import torch
from .block_lazy_tensor import BlockLazyTensor
from ..utils.getitem import _noop_index


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
        other = other.contiguous().view(*shape).expand(*expand_shape)
        return other

    """
    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        if self.num_blocks is None:
            train_train_covar_inv_root = train_train_covar_inv_root.unsqueeze(0)
            train_train_covar_inv_root = train_train_covar_inv_root.expand(
                self.base_lazy_tensor.size(0), train_train_covar_inv_root.size(-2), train_train_covar_inv_root.size(-1)
            )
        else:
            train_train_covar_inv_root = train_train_covar_inv_root.repeat(self.num_blocks, 1, 1)
        return self.base_lazy_tensor._exact_predictive_covar_inv_quad_form_cache(
            train_train_covar_inv_root, test_train_covar.base_lazy_tensor
        )

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        # Here the precomputed cache is a list
        # where each component in the list is the precomputed cache for each component lazy tensor
        res = self.base_lazy_tensor._exact_predictive_covar_inv_quad_form_root(
            precomputed_cache, test_train_covar.base_lazy_tensor
        )
        if self.num_blocks is not None:
            res = res.view(-1, self.num_blocks, res.size(1), res.size(2))
            res = res.sum(1)
        else:
            res = res.sum(0)
        return res
    """

    def _getitem(self, row_col_are_absorbed, row_index, col_index, *batch_indices):
        # Weird behavior can happen if any of the batch indices are tensors, so we'll use the default behavior
        if any(torch.is_tensor(batch_index) for batch_index in batch_indices) and len(batch_indices):
            return super(SumBatchLazyTensor, self)._getitem(row_col_are_absorbed, row_index, col_index, *batch_indices)

        res = self.base_lazy_tensor._getitem(row_col_are_absorbed, row_index, col_index, *batch_indices, _noop_index)
        if row_col_are_absorbed:
            block_dim = -3
            if torch.is_tensor(row_index):
                block_dim += 1
            if torch.is_tensor(col_index):
                block_dim += 1
            return res.sum(block_dim)
        else:
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
