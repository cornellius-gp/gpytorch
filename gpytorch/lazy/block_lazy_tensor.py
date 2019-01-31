#!/usr/bin/env python3

import torch
from .lazy_tensor import LazyTensor
from abc import abstractmethod


class BlockLazyTensor(LazyTensor):
    """
    An abstract LazyTensor class for block tensors.
    Super classes will determine how the different blocks are layed out
    (e.g. block diagonal, sum over blocks, etc.)

    BlockLazyTensors represent the groups of blocks as a batched Tensor.
    The :attr:`block_dim` attribute specifies which dimension of the base LazyTensor
    specifies the blocks.
    For example, (with `block_dim=-3` a `k x n x n` tensor represents `k` `n x n` blocks.
    A `b x k x n x n` tensor represents `k` `b x n x n` blocks.

    Args:
        - :attr:`base_lazy_tensor` (LazyTensor):
            A `k x n x n` LazyTensor, or a `b x k x n x n` LazyTensor.
        - :attr:`block_dim` (int):
            The dimension that specifies blocks.
    """

    def __init__(self, base_lazy_tensor, block_dim=-3):
        if base_lazy_tensor.dim() < 3:
            raise RuntimeError(
                "base_lazy_tensor must be a batch matrix (i.e.l at least 3 dimensions - got "
                "{}".format(base_lazy_tensor.dim())
            )

        # Make sure block_dim is positive
        block_dim = block_dim if block_dim < 0 else (block_dim - base_lazy_tensor.dim())

        super(BlockLazyTensor, self).__init__(base_lazy_tensor, block_dim=block_dim)
        self.base_lazy_tensor = base_lazy_tensor
        self.block_dim = block_dim

    @property
    def _positive_block_dim(self):
        """
        The block dimension - in positive number format
        """
        return self.base_lazy_tensor.dim() + self.block_dim

    @abstractmethod
    def _add_batch_dim(self, other):
        raise NotImplementedError

    def _expand_batch(self, batch_shape):
        batch_shape = list(batch_shape)
        batch_shape.insert(self._positive_block_dim, self.base_lazy_tensor.size(self._positive_block_dim))
        batch_shape = torch.Size(batch_shape)
        return self.__class__(self.base_lazy_tensor._expand_batch(batch_shape), block_dim=self.block_dim)

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

    def _unsqueeze_batch(self, dim):
        if dim > self._positive_block_dim:
            base_lazy_tensor = self.base_lazy_tensor._unsqueeze_batch(dim + 1)
            block_dim = self._positive_block_dim
        else:
            base_lazy_tensor = self.base_lazy_tensor._unsqueeze_batch(dim)
            block_dim = self.block_dim
        res = self.__class__(base_lazy_tensor, block_dim=block_dim)
        return res

    @abstractmethod
    def _remove_batch_dim(self, other):
        raise NotImplementedError

    def _transpose_nonbatch(self):
        return self.__class__(self.base_lazy_tensor._transpose_nonbatch(), block_dim=self.block_dim)

    def mul(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the block structure

        if not (torch.is_tensor(other) or isinstance(other, LazyTensor)) or (
            torch.is_tensor(other) and other.numel() == 1
        ):
            from .constant_mul_lazy_tensor import ConstantMulLazyTensor

            return self.__class__(ConstantMulLazyTensor(self.base_lazy_tensor, other), block_dim=self.block_dim)
        else:
            return super(BlockLazyTensor, self).mul(other)

    def zero_mean_mvn_samples(self, num_samples):
        res = self.base_lazy_tensor.zero_mean_mvn_samples(num_samples)
        res = self._remove_batch_dim(res.unsqueeze(-1)).squeeze(-1)
        return res
