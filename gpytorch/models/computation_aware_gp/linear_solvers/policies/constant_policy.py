#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch

from linear_operator.operators import LinearOperator

from ._sparsify_vector import sparsify_vector
from .linear_solver_policy import LinearSolverPolicy


class ConstantPolicy(LinearSolverPolicy):
    def __init__(self, actions: torch.Tensor) -> None:
        self.actions = actions

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            # Custom gradient
            action = self.actions[:, solver_state.iteration]

            # # Sparsify
            # if self.num_nonzero is not None:
            #     action = sparsify_vector(action, num_non_zero=self.num_nonzero)

            return action
