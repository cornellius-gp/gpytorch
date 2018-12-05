#!/usr/bin/env python3

import torch
from .block_lazy_tensor import BlockLazyTensor
from .. import settings


class SumBatchLazyTensor(BlockLazyTensor):
    """
    Represents a lazy tensor that is actually the sum of several lazy tensors blocks.
    For example, a `k x n x n` tensor represents `k` `n x n` blocks.
    Therefore, all the block diagonal components must be the same lazy tensor
    type and size.

    For a SumBatchLazyTensor in batch mode, the batch dimension is used to represent
    the actual ("true") batches as well as the different blocks.
    For example, `k` `b x n x n` blocks would be represented as a `bk x n x n`
    Tensor, where the "outer" batch dimension represents the true batch dimension
    (i.e. - the Tensor could be viewed as a `b x k x n x n` Tensor without re-ordering).

    For batch mode, the :attr:`groups` attribute specifes the number of blocks (to differentiate
    from true batches). This attribute should be `None` for non-batched Tensors.

    Args:
        :attr:`base_lazy_tensor` (LazyTensor):
            A `k x n x n` LazyTensor, or a `bk x n x n` LazyTensor, representing `k` blocks.
        :attr:`groups` (int or None):
            Set this to `k` for `bk x n x n` batched LazyTensors, or `None` for `k x n x n`
            unbatched LazyTensors.
    """

    def _matmul(self, rhs):
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        if self.num_blocks is None:
            rhs = rhs.unsqueeze(0)
            rhs = rhs.expand(self.base_lazy_tensor.size(0), *tuple(rhs.size())[1:])
        else:
            rhs = rhs.unsqueeze(1)
            rhs = rhs.expand(rhs.size(0), self.num_blocks, *tuple(rhs.size())[2:])
            rhs = rhs.contiguous().view(-1, rhs.size(-2), rhs.size(-1))

        res = self.base_lazy_tensor._matmul(rhs)

        if self.num_blocks is not None:
            res = res.view(-1, self.num_blocks, res.size(1), res.size(2))
        res = res.sum(-3)

        if isvector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if self.num_blocks is None:
            left_vecs = left_vecs.unsqueeze(0)
            left_vecs = left_vecs.expand(self.base_lazy_tensor.size(0), *tuple(left_vecs.size())[1:])
            right_vecs = right_vecs.unsqueeze(0)
            right_vecs = right_vecs.expand(self.base_lazy_tensor.size(0), *tuple(right_vecs.size())[1:])
        else:
            left_vecs = left_vecs.unsqueeze(1)
            left_vecs = left_vecs.expand(left_vecs.size(0), self.num_blocks, *tuple(left_vecs.size())[2:])
            left_vecs = left_vecs.contiguous().view(-1, left_vecs.size(-2), left_vecs.size(-1))
            right_vecs = right_vecs.unsqueeze(1)
            right_vecs = right_vecs.expand(right_vecs.size(0), self.num_blocks, *tuple(right_vecs.size())[2:])
            right_vecs = right_vecs.contiguous().view(-1, right_vecs.size(-2), right_vecs.size(-1))

        res = self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        return res

    def _size(self):
        base_size = self.base_lazy_tensor.size()
        if self.num_blocks is None:
            return torch.Size(list(base_size)[1:])
        else:
            inner_batch_size = self.base_lazy_tensor.size(0) // self.num_blocks
            return torch.Size([inner_batch_size] + list(base_size)[1:])

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        if self.num_blocks is None:
            if settings.debug.on():
                assert len(batch_indices) == 0
            batch_indices = torch.arange(0, self.base_lazy_tensor.size(0), dtype=torch.long, device=left_indices.device)
            batch_indices = batch_indices.unsqueeze(1).repeat(1, len(left_indices)).view(-1)
            left_indices = left_indices.unsqueeze(1).repeat(self.base_lazy_tensor.size(0), 1).view(-1)
            right_indices = right_indices.unsqueeze(1).repeat(self.base_lazy_tensor.size(0), 1).view(-1)
            res = self.base_lazy_tensor._get_indices(left_indices, right_indices, batch_indices)
            return res.view(self.base_lazy_tensor.size(0), -1).sum(0)
        else:
            if settings.debug.on():
                assert len(batch_indices) == 1
            batch_indices = batch_indices[0]
            inner_batch_indices = torch.arange(0, self.num_blocks, dtype=torch.long, device=left_indices.device)
            inner_batch_indices.unsqueeze_(1)
            batch_indices = batch_indices.mul(self.num_blocks).unsqueeze_(0).add(inner_batch_indices).view(-1)
            left_indices = left_indices.unsqueeze(0).repeat(self.num_blocks, 1).view(-1)
            right_indices = right_indices.unsqueeze(0).repeat(self.num_blocks, 1).view(-1)
            res = self.base_lazy_tensor._get_indices(left_indices, right_indices, batch_indices)

            res = self.base_lazy_tensor._get_indices(left_indices, right_indices, batch_indices)
            res = res.view(self.num_blocks, -1).sum(0)
            return res

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

    def diag(self):
        diag = self.base_lazy_tensor.diag()
        if self.num_blocks is not None:
            diag = diag.view(-1, self.num_blocks, diag.size(-1))
        return diag.sum(-2)

    def zero_mean_mvn_samples(self, num_samples):
        num_dim = self.size(-2)
        res = self.base_lazy_tensor.zero_mean_mvn_samples(num_samples)
        if self.num_blocks is None:
            res = res.view(num_samples, -1, num_dim).sum(1)
        else:
            res = res.view(num_samples, -1, self.num_blocks, num_dim).sum(2)
        return res
