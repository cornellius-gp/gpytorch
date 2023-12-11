#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import torch

from .linear_solver_policy import LinearSolverPolicy


class RademacherPolicy(LinearSolverPolicy):
    """Policy choosing randomized actions with entries drawn iid from a Rademacher distribution."""

    def __init__(
        self,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        action = torch.zeros(
            solver_state.problem.A.shape[0],
            dtype=solver_state.problem.A.dtype,
            device=solver_state.problem.A.device,
        )

        if self.num_nonzero is None:
            num_nonzero = solver_state.problem.A.shape[0]
        else:
            num_nonzero = self.num_nonzero

        with torch.no_grad():
            rademacher_vec = torch.ones(
                num_nonzero,
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
            ) - 2.0 * torch.bernoulli(
                0.5
                * torch.ones(
                    num_nonzero,
                    dtype=solver_state.problem.A.dtype,
                    device=solver_state.problem.A.device,
                )
            )

            perm = torch.randperm(solver_state.problem.A.shape[0])
            idcs = perm[0:num_nonzero]

            action[idcs] = rademacher_vec

        return action
