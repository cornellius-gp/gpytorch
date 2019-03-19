#!/usr/bin/env python3

import torch
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify
from abc import abstractmethod


class BlockLazyTensor(LazyTensor):
    """
    An abstract LazyTensor class for block tensors.
    Super classes will determine how the different blocks are layed out
    (e.g. block diagonal, sum over blocks, etc.)

    BlockLazyTensors represent the groups of blocks as a batched Tensor.
    The :attr:block_dim` attribute specifies which dimension of the base LazyTensor
    specifies the blocks.
    For example, (with `block_dim=-3` a `k x n x n` tensor represents `k` `n x n` blocks.
    A `b x k x n x n` tensor represents `k` `b x n x n` blocks.

    Args:
        - :attr:`base_lazy_tensor` (LazyTensor or Tensor):
            Must be at least 3 dimenional.
        - :attr:`block_dim` (int):
            The dimension that specifies blocks.
    """

    def __init__(self, base_lazy_tensor, block_dim=-3):
        if base_lazy_tensor.dim() < 3:
            raise RuntimeError(
                "base_lazy_tensor must be a batch matrix (i.e. at least 3 dimensions - got "
                "{}".format(base_lazy_tensor.dim())
            )

        # Make sure block_dim is negative
        block_dim = block_dim if block_dim < 0 else (block_dim - base_lazy_tensor.dim())

        # Everything is MUCH easier to write if the last batch dimension is the block dimension
        # I.e. blopck_dim = -3
        # We'll permute the dimensions if this is not the case
        if block_dim != -3:
            positive_block_dim = base_lazy_tensor.dim() + block_dim
            base_lazy_tensor = base_lazy_tensor._permute_batch(
                *range(positive_block_dim), *range(positive_block_dim + 1, base_lazy_tensor.dim() - 2),
                positive_block_dim
            )

        super(BlockLazyTensor, self).__init__(lazify(base_lazy_tensor))
        self.base_lazy_tensor = base_lazy_tensor

    @abstractmethod
    def _add_batch_dim(self, other):
        raise NotImplementedError

    def _expand_batch(self, batch_shape):
        batch_shape = torch.Size((*batch_shape, self.base_lazy_tensor.size(-3)))
        res = self.__class__(self.base_lazy_tensor._expand_batch(batch_shape))
        return res

    def _matmul(self, rhs):
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        rhs = self._add_batch_dim(rhs)
        res = self.base_lazy_tensor._matmul(rhs)
        res = self._remove_batch_dim(res)

        if isvector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        left_vecs = self._add_batch_dim(left_vecs)
        right_vecs = self._add_batch_dim(right_vecs)
        res = self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        return res

    def _permute_batch(self, *dims):
        if torch.is_tensor(self.base_lazy_tensor):
            base_lazy_tensor = self.base_lazy_tensor.permute(*dims, -3, -2, -1)
        else:
            base_lazy_tensor = self.base_lazy_tensor._permute_batch(*dims, self.base_lazy_tensor.dim() - 3)
        res = self.__class__(base_lazy_tensor)
        return res

    def _unsqueeze_batch(self, dim):
        if torch.is_tensor(self.base_lazy_tensor):
            base_lazy_tensor = self.base_lazy_tensor.unsqueeze(dim)
        else:
            base_lazy_tensor = self.base_lazy_tensor._unsqueeze_batch(dim)
        res = self.__class__(base_lazy_tensor)
        return res

    @abstractmethod
    def _remove_batch_dim(self, other):
        raise NotImplementedError

    def _mul_constant(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the block structure
        from .constant_mul_lazy_tensor import ConstantMulLazyTensor
        return self.__class__(ConstantMulLazyTensor(self.base_lazy_tensor, other))

    def _transpose_nonbatch(self):
        return self.__class__(self.base_lazy_tensor._transpose_nonbatch())

    def zero_mean_mvn_samples(self, num_samples):
        res = self.base_lazy_tensor.zero_mean_mvn_samples(num_samples)
        res = self._remove_batch_dim(res.unsqueeze(-1)).squeeze(-1)
        return res
