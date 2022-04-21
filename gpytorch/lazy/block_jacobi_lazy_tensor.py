#!/usr/bin/env python3

from abc import abstractmethod

import torch

from ..utils.memoize import cached

from .block_diag_lazy_tensor import BlockDiagLazyTensor
from .non_lazy_tensor import lazify


class BlockJacobiLazyTensor(BlockDiagLazyTensor):
    """
    Same as block diag lazy tensor, but the last block have a different size.
    """
    def __init__(self, batched_blocks, last_block):
        self.num_main_blocks = batched_blocks.size(-3)

        self.main_block_size = batched_blocks.size(-1)
        self.last_block_size = last_block.size(-1)

        zero_padding = torch.zeros(
            self.main_block_size - self.last_block_size, self.last_block_size,
            dtype=batched_blocks.dtype,
            device=batched_blocks.device,
        )
        eye = torch.eye(
            self.main_block_size - self.last_block_size,
            dtype=batched_blocks.dtype,
            device=batched_blocks.device,
        )

        first_row = torch.cat(
            (last_block, zero_padding.T),
            dim=-1
        )
        second_row = torch.cat(
            (zero_padding, eye),
            dim=-1
        )
        padded_last_block = torch.cat((first_row, second_row), dim=-2)

        concatenated_blocks = torch.cat(
            (batched_blocks, padded_last_block.unsqueeze(-3)),
            dim=-3,
        )
        super().__init__(lazify(concatenated_blocks))
        self._args = (batched_blocks, last_block)

        # BlockDiagLazyTensor(concatenated_blocks)

    @cached(name="cholesky")
    def _cholesky(self, upper=False):
        from .triangular_lazy_tensor import TriangularLazyTensor

        chol_blocks = self.base_lazy_tensor.cholesky(upper=upper)
        chol = BlockDiagLazyTensor(chol_blocks)
        return TriangularLazyTensor(chol, upper=upper)

    def diag(self):
        res = self.base_lazy_tensor.diag().contiguous()
        res = res.view(*self.batch_shape, -1)
        return res[..., 0:self.size(-1)]

    def inv_matmul(self, rhs):
        zero_padding = torch.zeros(
            self.main_block_size - self.last_block_size, rhs.size(-1),
            dtype=rhs.dtype,
            device=rhs.device,
        )
        padded_rhs = torch.cat(
            (rhs, zero_padding),
            dim=-2,
        )
        res = super().inv_matmul(padded_rhs)
        return res[..., 0:self.size(-2), :]

    def logdet(self):
        return self.diag().abs().log().sum()

    def _matmul(self, rhs):
        zero_padding = torch.zeros(
            self.main_block_size - self.last_block_size, rhs.size(-1),
            dtype=rhs.dtype,
            device=rhs.device,
        )
        padded_rhs = torch.cat(
            (rhs, zero_padding),
            dim=-2,
        )
        res = super()._matmul(padded_rhs)
        return res[..., 0:self.size(-2), :]

    def _size(self):
        n = self.num_main_blocks * self.main_block_size + self.last_block_size
        return n, n

    def _transpose_nonbatch(self):
        return self
