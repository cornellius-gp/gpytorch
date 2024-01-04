#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import torch
from linear_operator.operators import BlockSparseLinearOperator, LinearOperator

from ._sparsify_vector import sparsify_vector

from .linear_solver_policy import LinearSolverPolicy


class RademacherPolicy(LinearSolverPolicy):
    """Policy choosing randomized actions with entries drawn iid from a Rademacher distribution."""

    def __init__(
        self,
        num_non_zero: Optional[int] = None,
        non_zero_idcs: Optional[torch.Tensor] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> None:
        if num_non_zero is not None and non_zero_idcs is not None:
            if num_non_zero != non_zero_idcs.shape[-1]:
                raise ValueError(
                    f"Number of non-zero entries {num_non_zero} does not match the "
                    + f"non-zero index tensor of shape {non_zero_idcs.shape}."
                )
        if num_non_zero is None and non_zero_idcs is not None:
            num_non_zero = non_zero_idcs.shape[-1]

        self.num_nonzero = num_non_zero
        self.non_zero_idcs = non_zero_idcs
        self.preconditioner = preconditioner
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        action = torch.zeros(
            solver_state.problem.A.shape[0],
            dtype=solver_state.problem.A.dtype,
            device=solver_state.problem.A.device,
        )

        if self.num_nonzero is None or self.preconditioner is not None:
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

            if self.preconditioner is not None:
                action = self.preconditioner(rademacher_vec)
                return sparsify_vector(action, num_non_zero=self.num_nonzero)

            if self.non_zero_idcs is None:
                perm = torch.randperm(solver_state.problem.A.shape[0])
                non_zero_idcs = perm[0:num_nonzero]
            elif self.non_zero_idcs.ndim == 1:
                non_zero_idcs = self.non_zero_idcs.reshape(1, -1)
            else:
                non_zero_idcs = self.non_zero_idcs[solver_state.iteration, :].reshape(1, -1)

            if self.num_nonzero is None and self.non_zero_idcs is None:
                action[non_zero_idcs] = rademacher_vec
            else:
                action = BlockSparseLinearOperator(
                    non_zero_idcs=non_zero_idcs, blocks=rademacher_vec, size_sparse_dim=solver_state.problem.A.shape[0]
                )

        return action
