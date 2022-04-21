#!/usr/bin/env python3

from abc import abstractmethod

import torch

from ..utils.memoize import cached

from .lazy_tensor import LazyTensor
from .triangular_lazy_tensor import TriangularLazyTensor
from .block_diag_lazy_tensor import BlockDiagLazyTensor
from .non_lazy_tensor import lazify


class BlockTriangularLazyTensor(LazyTensor):
    """
    RTFS.
    """
    def __init__(self, batched_blocks, last_block, upper=False):
        super().__init__(batched_blocks, last_block, upper=upper)

        self.upper = upper

        self.num_main_blocks = batched_blocks.size(-3)
        self.main_block_size = batched_blocks.size(-1)
        self.last_block_size = last_block.size(-1)

        self.batched_blocks = batched_blocks
        self.last_block = last_block

    def diag(self):
        first_diag = lazify(self.batched_blocks).diag().contiguous().view(-1)
        second_diag = self.last_block.diag()

        return torch.cat((first_diag, second_diag), dim=-1)

    def inv_matmul(self, rhs):
        _, num_cols = rhs.size()

        first_rhs = rhs[0:self.num_main_blocks * self.main_block_size, :].view(
            self.num_main_blocks, self.main_block_size, -1,
        )
        # use reshape instead of view -- somehow the solution is not stored contiguously
        first_inv = torch.linalg.solve_triangular(
            self.batched_blocks, first_rhs, upper=self.upper,
        ).reshape(-1, num_cols)

        second_rhs = rhs[self.num_main_blocks * self.main_block_size:, :]
        second_inv = torch.linalg.solve_triangular(
            self.last_block, second_rhs, upper=self.upper,
        )

        """
        Maybe call .contiguous()?
        """
        return torch.cat((first_inv, second_inv), dim=-2)

    def logdet(self):
        return self.diag().abs().log().sum()

    def _matmul(self, rhs):
        _, num_cols = rhs.size()

        first_rhs = rhs[0:self.num_main_blocks * self.main_block_size, :].view(
            self.num_main_blocks, self.main_block_size, -1,
        )
        first_mvm = self.batched_blocks.matmul(first_rhs).view(-1, num_cols)

        second_rhs = rhs[self.num_main_blocks * self.main_block_size:, :]
        second_mvm = self.last_block.matmul(second_rhs)

        """
        Maybe call .contiguous()?
        """
        return torch.cat((first_mvm, second_mvm), dim=-2)

    def _size(self):
        n = self.num_main_blocks * self.main_block_size + self.last_block_size
        return n, n

    def _transpose_nonbatch(self):
        return BlockTriangularLazyTensor(
            self.batched_blocks.transpose(-2, -1),
            self.last_block.transpose(-2, -1),
            upper=not self.upper,
        )
