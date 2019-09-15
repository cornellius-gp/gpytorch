#!/usr/bin/env python3

import torch
from .block_lazy_tensor import BlockLazyTensor
from ..utils.memoize import cached


class BlockDiagLazyTensor(BlockLazyTensor):
    """
    Represents a lazy tensor that is the block diagonal of square matrices.
    The :attr:`block_dim` attribute specifies which dimension of the base LazyTensor
    specifies the blocks.
    For example, (with `block_dim=-3` a `k x n x n` tensor represents `k` `n x n` blocks (a `kn x kn` matrix).
    A `b x k x n x n` tensor represents `k` `b x n x n` blocks (a `b x kn x kn` batch matrix).

    Args:
        :attr:`base_lazy_tensor` (LazyTensor or Tensor):
            Must be at least 3 dimensional.
        :attr:`block_dim` (int):
            The dimension that specifies the blocks.
    """
    @property
    def num_blocks(self):
        return self.base_lazy_tensor.size(-3)

    def _add_batch_dim(self, other):
        *batch_shape, num_rows, num_cols = other.shape
        batch_shape = list(batch_shape)

        batch_shape.append(self.num_blocks)
        other = other.view(*batch_shape, num_rows // self.num_blocks, num_cols)
        return other

    @cached(name="cholesky")
    def _cholesky(self):
        return self.__class__(self.base_lazy_tensor._cholesky())

    def _cholesky_solve(self, rhs):
        rhs = self._add_batch_dim(rhs)
        res = self.base_lazy_tensor._cholesky_solve(rhs)
        res = self._remove_batch_dim(res)
        return res

    def _get_indices(self, row_index, col_index, *batch_indices):
        # Figure out what block the row/column indices belong to
        row_index_block = row_index.div(self.base_lazy_tensor.size(-2))
        col_index_block = col_index.div(self.base_lazy_tensor.size(-1))

        # Find the row/col index within each block
        row_index = row_index.fmod(self.base_lazy_tensor.size(-2))
        col_index = col_index.fmod(self.base_lazy_tensor.size(-1))

        # If the row/column blocks do not agree, then we have off diagonal elements
        # These elements should be zeroed out
        res = self.base_lazy_tensor._get_indices(row_index, col_index, *batch_indices, row_index_block)
        res = res * torch.eq(row_index_block, col_index_block).type_as(res)
        return res

    def _remove_batch_dim(self, other):
        shape = list(other.shape)
        del shape[-3]
        shape[-2] *= self.num_blocks
        other = other.reshape(*shape)
        return other

    def _root_decomposition(self):
        return self.__class__(self.base_lazy_tensor._root_decomposition())

    def _root_inv_decomposition(self, initial_vectors=None):
        return self.__class__(self.base_lazy_tensor._root_inv_decomposition(initial_vectors))

    def _size(self):
        shape = list(self.base_lazy_tensor.shape)
        shape[-2] *= shape[-3]
        shape[-1] *= shape[-3]
        del shape[-3]
        return torch.Size(shape)

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        if num_tridiag:
            return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)
        else:
            rhs = self._add_batch_dim(rhs)
            res = self.base_lazy_tensor._solve(rhs, preconditioner, num_tridiag=None)
            res = self._remove_batch_dim(res)
            return res

    def diag(self):
        res = self.base_lazy_tensor.diag().contiguous()
        return res.view(*self.batch_shape, self.size(-1))

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if inv_quad_rhs is not None:
            inv_quad_rhs = self._add_batch_dim(inv_quad_rhs)
        inv_quad_res, logdet_res = self.base_lazy_tensor.inv_quad_logdet(
            inv_quad_rhs, logdet, reduce_inv_quad=reduce_inv_quad
        )
        if inv_quad_res is not None and inv_quad_res.numel():
            if reduce_inv_quad:
                inv_quad_res = inv_quad_res.view(*self.base_lazy_tensor.batch_shape)
                inv_quad_res = inv_quad_res.sum(-1)
            else:
                inv_quad_res = inv_quad_res.view(*self.base_lazy_tensor.batch_shape, inv_quad_res.size(-1))
                inv_quad_res = inv_quad_res.sum(-2)
        if logdet_res is not None and logdet_res.numel():
            logdet_res = logdet_res.view(*logdet_res.shape).sum(-1)
        return inv_quad_res, logdet_res
