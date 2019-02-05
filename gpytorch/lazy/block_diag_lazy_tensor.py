#!/usr/bin/env python3

import torch

from .. import settings
from ..utils.memoize import cached
from .block_lazy_tensor import BlockLazyTensor
from .non_lazy_tensor import NonLazyTensor
from .root_lazy_tensor import RootLazyTensor


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
        return self.base_lazy_tensor.size(self.block_dim)

    def _add_batch_dim(self, other):
        *batch_shape, num_rows, num_cols = other.shape
        batch_shape = list(batch_shape)

        if self.block_dim == -3:
            batch_shape.append(self.num_blocks)
        else:
            insert_dim = self.block_dim + 3
            batch_shape.insert(insert_dim, self.num_blocks)
        other = other.contiguous().view(*batch_shape, num_rows // self.num_blocks, num_cols)
        return other

    def _remove_batch_dim(self, other):
        shape = list(other.shape)

        del shape[self.block_dim]
        shape[-2] *= self.num_blocks
        other = other.contiguous().view(*shape)
        return other

    def _size(self):
        shape = list(self.base_lazy_tensor.shape)
        shape[-2] *= shape[self.block_dim]
        shape[-1] *= shape[self.block_dim]
        del shape[self.block_dim]
        return torch.Size(shape)

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
                inv_quad_res = inv_quad_res.sum(self._positive_block_dim)
            else:
                inv_quad_res = inv_quad_res.view(*self.base_lazy_tensor.batch_shape, inv_quad_res.size(-1))
                inv_quad_res = inv_quad_res.sum(self._positive_block_dim)
        if logdet_res is not None and logdet_res.numel():
            logdet_res = logdet_res.view(*logdet_res.shape).sum(self._positive_block_dim)
        return inv_quad_res, logdet_res

    @cached(name="root_decomposition")
    def root_decomposition(self):
        if settings.fast_computations.covar_root_decomposition.on():
            res = self.__class__(self.base_lazy_tensor.root_decomposition().root, block_dim=self.block_dim)
        else:
            chol = torch.cholesky(self.base_lazy_tensor.evaluate())
            res = self.__class__(NonLazyTensor(chol), block_dim=self.block_dim)
        return RootLazyTensor(res)

    def zero_mean_mvn_samples(self, sample_shape=torch.Size()):
        res = self.base_lazy_tensor.zero_mean_mvn_samples(sample_shape=sample_shape)

        # Move the block dimension to the appropriate place
        res = res.unsqueeze(-2).transpose(-2, self.block_dim).squeeze(self.block_dim).contiguous()
        res = res.view(*sample_shape, *self.batch_shape, self.size(-2))
        return res
