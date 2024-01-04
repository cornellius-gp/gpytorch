#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch
from linear_operator.operators import BlockSparseLinearOperator, LinearOperator

from .linear_solver_policy import LinearSolverPolicy


class StochasticGradientPolicy(LinearSolverPolicy):
    """Policy choosing stochastic gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions."""

    def __init__(
        self, batch_size: int = 64, preconditioner: Optional[LinearOperator] = None, dense: bool = False
    ) -> None:
        self.num_nonzero = batch_size
        self.preconditioner = preconditioner
        self.dense = dense

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            residual = solver_state.residual

            if isinstance(self.preconditioner, (torch.Tensor, LinearOperator)):
                residual = self.preconditioner @ residual
            elif callable(self.preconditioner):
                residual = self.preconditioner(residual).squeeze()

            # Sample non-zero indices (i.e. batches to compute gradients for)
            perm = torch.randperm(residual.shape[0])
            non_zero_idcs = perm[0 : self.num_nonzero]

            if self.dense:
                # Stochastic gradient (residual)
                action = torch.zeros(
                    residual.shape[0],
                    dtype=residual.dtype,
                    device=residual.device,
                )
                action[non_zero_idcs] = residual[non_zero_idcs]
                return action

            else:
                return BlockSparseLinearOperator(
                    non_zero_idcs=non_zero_idcs, blocks=residual[non_zero_idcs], size_sparse_dim=len(perm)
                )
