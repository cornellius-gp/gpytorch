#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch
from linear_operator import utils

from linear_operator.operators import LinearOperator

from ._sparsify_vector import sparsify_vector
from .linear_solver_policy import LinearSolverPolicy


class GradientPolicy(LinearSolverPolicy):
    """Policy choosing (preconditioned) gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions.

    :param reorthogonalization_period: If not None, reorthogonalize every given period of iterations.
    :param precond: Preconditioner :math:`P \\approx A^{-1}`.
    """

    def __init__(
        self,
        reorthogonalization_period: Optional[int] = 20,
        preconditioner: Optional["LinearOperator"] = None,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.reorthogonalization_period = reorthogonalization_period
        self.preconditioner = preconditioner
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            action = solver_state.residual

            # Reorthogonalization
            if self.reorthogonalization_period is not None and solver_state.iteration > 0:
                if (solver_state.iteration % self.reorthogonalization_period) == 0:
                    # Reorthogonalize actions twice for stability: https://doi.org/10.1007/s00211-005-0615-4
                    actions, _ = torch.linalg.qr(
                        torch.vstack((solver_state.cache["actions_op"], torch.atleast_2d(action))).mT, mode="reduced"
                    )
                    actions, _ = torch.linalg.qr(actions, mode="reduced")

                    # Update actions
                    solver_state.cache["actions_op"] = actions[:, 0:-1].mT

                    # Update Gramian
                    gram_SKhatS = solver_state.cache["actions_op"] @ (
                        solver_state.problem.A @ solver_state.cache["actions_op"].mT
                    )

                    # Update Cholesky factor
                    solver_state.cache["cholfac_gram"] = utils.cholesky.psd_safe_cholesky(gram_SKhatS, upper=False)

                    # Update
                    action = actions[:, -1]

            # Preconditioning
            if isinstance(self.preconditioner, (torch.Tensor, LinearOperator)):
                action = self.preconditioner @ action
            elif callable(self.preconditioner):
                action = self.preconditioner(action).squeeze()

            # Sparsify
            if self.num_nonzero is not None:
                action = sparsify_vector(action, num_non_zero=self.num_nonzero)

            return action
