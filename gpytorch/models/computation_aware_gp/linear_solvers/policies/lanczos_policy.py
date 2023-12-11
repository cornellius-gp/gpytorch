#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import torch

from linear_operator.operators import LinearOperator
from linear_operator.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

from ._sparsify_vector import sparsify_vector
from .linear_solver_policy import LinearSolverPolicy


class LanczosPolicy(LinearSolverPolicy):
    """Policy choosing approximate eigenvectors as actions."""

    def __init__(
        self,
        seeding: float = "random",
        precond: Optional["LinearOperator"] = None,
        sparsification_threshold: float = 0.0,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.seeding = seeding
        self.precond = precond
        self.sparsification_threshold = sparsification_threshold
        self.num_nonzero = num_non_zero
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            if "init_vec" not in solver_state.cache:
                # Seed vector
                if self.seeding == "random":
                    init_vec = torch.randn(
                        solver_state.problem.A.shape[1],
                        dtype=solver_state.problem.A.dtype,
                        device=solver_state.problem.A.device,
                    )
                    init_vec = init_vec.div(torch.linalg.vector_norm(init_vec))
                else:
                    raise NotImplementedError

                # Cache initial vector
                solver_state.cache["init_vec"] = init_vec

            if "compressed_solution" in solver_state.cache:
                if solver_state.cache["compressed_solution"] is None:
                    action = solver_state.cache["init_vec"]
                else:
                    action = (
                        solver_state.cache["init_vec"] - solver_state.cache["linear_op_actions_compressed_solution"]
                    )
            else:
                action = solver_state.cache["init_vec"] - solver_state.problem.A @ solver_state.solution

            if isinstance(self.precond, (torch.Tensor, LinearOperator)):
                action = self.precond @ action
            elif callable(self.precond):
                action = self.precond(action).squeeze()

            if self.sparsification_threshold > 0.0:
                action[torch.abs(action) < self.sparsification_threshold] = 0.0

            # Sparsify
            if self.num_nonzero is not None:
                action = sparsify_vector(action, num_non_zero=self.num_nonzero)

            return action


class SubsetLanczosPolicy(LinearSolverPolicy):
    """Policy choosing approximate eigenvectors as actions."""

    def __init__(self, subset_size: int = 256) -> None:
        self.subset_size = subset_size
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        if "init_vec" not in solver_state.cache:
            # Seed vector
            init_vec = torch.randn(
                self.subset_size,
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
            )
            init_vec = init_vec.div(torch.linalg.vector_norm(init_vec))

            # Cache initial vector
            solver_state.cache["init_vec"] = init_vec

        action = torch.zeros(
            solver_state.problem.A.shape[1],
            dtype=solver_state.problem.A.dtype,
            device=solver_state.problem.A.device,
        )

        action[0 : self.subset_size] = (
            solver_state.cache["init_vec"]
            - solver_state.problem.A[0 : self.subset_size, 0 : self.subset_size]
            @ solver_state.solution[0 : self.subset_size]
        )

        return action


class FullLanczosPolicy(LinearSolverPolicy):
    """Policy choosing approximate eigenvectors as actions."""

    def __init__(self, descending: bool = True, max_iter: Optional[int] = None) -> None:
        self.descending = descending
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        if solver_state.iteration == 0:
            # Compute approximate eigenvectors via Lanczos process

            # Initial seed vector
            init_vecs = solver_state.residual.unsqueeze(-1)
            init_vecs = torch.randn(
                solver_state.problem.A.shape[1],
                1,
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
            )

            # Lanczos tridiagonalization
            Q, T = lanczos_tridiag(
                solver_state.problem.A.matmul,
                init_vecs=init_vecs,
                max_iter=solver_state.problem.A.shape[1] if self.max_iter is None else self.max_iter,
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
                matrix_shape=solver_state.problem.A.shape,
                tol=1e-5,
            )
            evals_lanczos, evecs_T = lanczos_tridiag_to_diag(T)
            evecs_lanczos = Q @ evecs_T

            # Cache approximate eigenvectors
            solver_state.cache["evals_lanczos"], idcs = torch.sort(evals_lanczos, descending=self.descending)
            solver_state.cache["evecs_lanczos"] = evecs_lanczos[:, idcs]

            # Cache initial vector
            solver_state.cache["init_vec"] = init_vecs.squeeze(-1).div(torch.linalg.vector_norm(init_vecs))

        # Return approximate eigenvectors according to strategy
        if solver_state.iteration < solver_state.cache["evecs_lanczos"].shape[1]:
            return solver_state.cache["evecs_lanczos"][:, solver_state.iteration]
        else:
            return solver_state.residual
