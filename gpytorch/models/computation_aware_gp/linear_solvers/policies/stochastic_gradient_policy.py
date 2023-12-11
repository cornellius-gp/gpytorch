#!/usr/bin/env python3

from __future__ import annotations

import torch

from .linear_solver_policy import LinearSolverPolicy


class StochasticGradientPolicy(LinearSolverPolicy):
    """Policy choosing stochastic gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions."""

    def __init__(
        self,
        num_non_zero: int = 64,
    ) -> None:
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            # Sample non-zero indices (i.e. batches to compute gradients for)
            perm = torch.randperm(solver_state.residual.shape[0])
            non_zero_idcs = perm[0 : self.num_nonzero]

            # Stochastic gradient (residual)
            action = torch.zeros(
                solver_state.residual.shape[0], dtype=solver_state.residual.dtype, device=solver_state.residual.device
            )

            action[non_zero_idcs] = solver_state.residual[non_zero_idcs]

            return action
