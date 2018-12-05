#!/usr/bin/env python3

import torch
from .block_lazy_tensor import BlockLazyTensor
from .. import settings


class BlockDiagLazyTensor(BlockLazyTensor):
    """
    Represents a lazy tensor that is the block diagonal of square matrices.
    For example, a `k x n x n` tensor represents `k` `n x n` blocks.
    Therefore, all the block diagonal components must be the same lazy tensor
    type and size.

    For a BlockDiagLazyTensor in batch mode, the batch dimension is used to represent
    the actual ("true") batches as well as the different blocks.
    For example, `k` `b x n x n` blocks would be represented as a `bk x n x n`
    Tensor, where the "outer" batch dimension represents the true batch dimension
    (i.e. - the Tensor could be viewed as a `b x k x n x n` Tensor without re-ordering).

    For batch mode, the :attr:`num_blocks` attribute specifes the number of blocks (to differentiate
    from true batches). This attribute should be `None` for non-batched Tensors.

    Args:
        :attr:`base_lazy_tensor` (LazyTensor):
            A `k x n x n` LazyTensor, or a `bk x n x n` LazyTensor, representing `k` blocks.
        :attr:`num_blocks` (int or None):
            Set this to `k` for `bk x n x n` batched LazyTensors, or `None` for `k x n x n`
            unbatched LazyTensors.
    """

    def _matmul(self, rhs):
        block_size = self.base_lazy_tensor.size(-1)
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        num_cols = rhs.size(-1)
        rhs = rhs.contiguous().view(-1, block_size, num_cols)

        res = self.base_lazy_tensor._matmul(rhs)

        if self.num_blocks is not None:
            res = res.contiguous().view(-1, self.num_blocks * res.size(1), res.size(2))
        else:
            res = res.contiguous().view(res.size(0) * res.size(1), res.size(2))

        if isvector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        block_size = self.base_lazy_tensor.size(-1)
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        left_vecs = left_vecs.view(-1, block_size, left_vecs.size(-1))
        right_vecs = right_vecs.view(-1, block_size, right_vecs.size(-1))
        res = self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        return res

    def _size(self):
        base_size = self.base_lazy_tensor.size()
        if self.num_blocks is None:
            return torch.Size((base_size[0] * base_size[1], base_size[0] * base_size[2]))
        else:
            true_batch_size = self.base_lazy_tensor.size(0) // self.num_blocks
            return torch.Size((true_batch_size, self.num_blocks * base_size[1], self.num_blocks * base_size[2]))

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        if self.num_blocks is None:
            if settings.debug.on():
                assert len(batch_indices) == 0
            block_size = self.base_lazy_tensor.size(-1)
            left_batch_indices = left_indices.div(block_size).long()
            right_batch_indices = left_indices.div(block_size).long()
            left_indices = left_indices.fmod(block_size)
            right_indices = right_indices.fmod(block_size)

            res = self.base_lazy_tensor._get_indices(left_indices, right_indices, left_batch_indices)
            res = res * torch.eq(left_batch_indices, right_batch_indices).type_as(res)
            return res
        else:
            if settings.debug.on():
                assert len(batch_indices) == 1
            batch_indices = batch_indices[0]
            block_size = self.base_lazy_tensor.size(-1)
            left_batch_indices = left_indices.div(block_size).long()
            right_batch_indices = left_indices.div(block_size).long()
            batch_indices = batch_indices * block_size + left_batch_indices
            left_indices = left_indices.fmod(block_size)
            right_indices = right_indices.fmod(block_size)

            res = self.base_lazy_tensor._get_indices(left_indices, right_indices, batch_indices)
            res = res * torch.eq(left_batch_indices, right_batch_indices).type_as(res)
            return res

    def diag(self):
        res = self.base_lazy_tensor.diag().contiguous()
        if self.num_blocks is not None:
            res = res.view(res.size(0) // self.num_blocks, -1)
        else:
            res = res.view(-1)
        return res

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):
        if inv_quad_rhs is not None:
            inv_quad_rhs = inv_quad_rhs.view(-1, self.base_lazy_tensor.size(-1), inv_quad_rhs.size(-1))
        inv_quad_res, log_det_res = self.base_lazy_tensor.inv_quad_log_det(
            inv_quad_rhs, log_det, reduce_inv_quad=reduce_inv_quad
        )
        if inv_quad_res is not None and inv_quad_res.numel():
            if reduce_inv_quad:
                inv_quad_res = inv_quad_res.view(*self.batch_shape, -1).sum(-1)
            else:
                inv_quad_res = inv_quad_res.view(*self.batch_shape, -1, inv_quad_res.size(-1)).sum(-2)
        if log_det_res is not None and log_det_res.numel():
            log_det_res = log_det_res.view(*self.batch_shape, -1).sum(-1)
        return inv_quad_res, log_det_res

    def zero_mean_mvn_samples(self, num_samples):
        res = self.base_lazy_tensor.zero_mean_mvn_samples(num_samples)
        if self.num_blocks is None:
            res = res.view(num_samples, -1)
        else:
            res = res.view(num_samples, self.base_lazy_tensor.size(0) // self.num_blocks, -1)
        return res
