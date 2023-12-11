#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch

from linear_operator.operators import LinearOperator

from ._sparsify_vector import sparsify_vector
from .linear_solver_policy import LinearSolverPolicy


class CustomGradientPolicy(LinearSolverPolicy):
    def __init__(
        self,
        linop: LinearOperator,
        rhs: torch.Tensor,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.linop = linop
        self.rhs = rhs
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            # Custom gradient
            action = self.rhs - self.linop @ solver_state.solution

            # Sparsify
            if self.num_nonzero is not None:
                action = sparsify_vector(action, num_non_zero=self.num_nonzero)

            return action
