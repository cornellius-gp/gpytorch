#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch

from linear_operator.operators import LinearOperator

from ._sparsify_vector import sparsify_vector
from .linear_solver_policy import LinearSolverPolicy


class GradientPolicy(LinearSolverPolicy):
    """Policy choosing (preconditioned) gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions.

    :param precond: Preconditioner :math:`P \\approx A^{-1}`.
    """

    def __init__(
        self,
        preconditioner: Optional["LinearOperator"] = None,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.preconditioner = preconditioner
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            action = solver_state.residual

            if isinstance(self.preconditioner, (torch.Tensor, LinearOperator)):
                action = self.preconditioner @ action
            elif callable(self.preconditioner):
                action = self.preconditioner(action).squeeze()

            # Sparsify
            if self.num_nonzero is not None:
                action = sparsify_vector(action, num_non_zero=self.num_nonzero)

            return action
