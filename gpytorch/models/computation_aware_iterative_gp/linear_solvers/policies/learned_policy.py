#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch
from linear_operator.operators import BlockDiagonalSparseLinearOperator

from .linear_solver_policy import LinearSolverPolicy


class LearnedPolicy(LinearSolverPolicy):
    """Policy choosing actions optimized alongside the hyperparameters of the kernel."""

    def __init__(self, non_zero_idcs: Optional[torch.Tensor] = None, blocks: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.non_zero_idcs = non_zero_idcs
        self.blocks = torch.nn.Parameter(blocks)

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        return BlockDiagonalSparseLinearOperator(
            non_zero_idcs=self.non_zero_idcs[solver_state.iteration, :],
            blocks=self.blocks[solver_state.iteration, :],
            size_sparse_dim=solver_state.problem.b.shape[0],
        )
