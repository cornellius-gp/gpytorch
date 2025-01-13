#!/usr/bin/env python3
from __future__ import annotations

import abc

import torch
from linear_operator.operators import BlockDiagonalSparseLinearOperator

from .linear_solver_policy import LinearSolverPolicy


class UnitVectorPolicy(LinearSolverPolicy):
    """Policy choosing unit vectors as actions.

    :param ordering: The order in which datapoints are selected via the corresponding unit entry
                     in the action. One of ["lexicographic", "max_abs_residual"].
    """

    class Ordering(abc.ABC):
        def __call__(self, solver_state: "LinearSolverState") -> int:
            raise NotImplementedError

    class Lexicographic(Ordering):
        def __call__(self, solver_state: "LinearSolverState") -> int:
            return solver_state.iteration

    class MaximumAbsoluteResidual(Ordering):
        def __call__(self, solver_state: "LinearSolverState") -> int:
            return torch.argmax(torch.abs(solver_state.residual)).item()

    def __init__(self, ordering=Lexicographic()) -> None:
        self.ordering = ordering
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        return BlockDiagonalSparseLinearOperator(
            non_zero_idcs=torch.as_tensor(
                self.ordering(solver_state), dtype=torch.long, device=solver_state.problem.A.device
            ),
            blocks=torch.as_tensor(1.0, dtype=solver_state.problem.A.dtype, device=solver_state.problem.A.device),
            size_sparse_dim=solver_state.problem.A.shape[1],
        )
