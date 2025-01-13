#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch

from linear_operator.operators import BlockDiagonalSparseLinearOperator

from ._sparsify_vector import sparsify_vector
from .linear_solver_policy import LinearSolverPolicy


class KernelJacobianPolicy(LinearSolverPolicy):
    """Policy choosing actions which estimate the spectrum of the kernel matrix Jacobian."""

    def __init__(
        self,
        num_non_zero: Optional[int] = None,
        non_zero_idcs: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_nonzero = num_non_zero
        self.non_zero_idcs = non_zero_idcs

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            if "seed_vector" not in solver_state.cache:
                # Seed vector
                seed_vector = torch.randn(
                    solver_state.problem.A.shape[1],
                    dtype=solver_state.problem.A.dtype,
                    device=solver_state.problem.A.device,
                )
                seed_vector = seed_vector.div(torch.linalg.vector_norm(seed_vector))

                # Cache seed vector
                solver_state.cache["seed_vector"] = seed_vector
                return seed_vector

        def rhs_minus_dlinop_dparams_times_solution_estimate(*representation):
            lin_op = solver_state.problem.A.representation_tree()(*representation)
            # TODO: enable batching by subsetting with non_zero_idcs
            actions_op = solver_state.cache["actions_op"]
            actions_op.requires_grad = False
            lin_op_actions = (
                (actions_op._matmul(lin_op)).mT
                if isinstance(actions_op, BlockDiagonalSparseLinearOperator)
                else lin_op @ actions_op.mT
            )
            return solver_state.cache["seed_vector"] - lin_op_actions @ solver_state.cache["compressed_solution"]

        action = torch.autograd.functional.jacobian(
            rhs_minus_dlinop_dparams_times_solution_estimate, solver_state.problem.A.representation()
        )[1].squeeze()

        with torch.no_grad():
            # Sparsify
            if self.num_nonzero is not None:
                action = sparsify_vector(action, num_non_zero=self.num_nonzero)

        return action
