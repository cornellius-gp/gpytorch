#!/usr/bin/env python3
from __future__ import annotations

import torch

from .linear_solver_policy import LinearSolverPolicy


class SpectralPolicy(LinearSolverPolicy):
    """Policy choosing eigenvectors as actions."""

    def __init__(self, descending: bool = True) -> None:
        self.descending = descending
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            if solver_state.iteration == 0:
                # Compute eigenvectors
                with torch.no_grad():
                    eigvals, eigvecs = torch.linalg.eigh(
                        solver_state.problem.A.to_dense() + 1e-4 * torch.eye(solver_state.problem.A.shape[0])
                    )

                # Cache eigenvectors
                solver_state.cache["eigvals"], idcs = torch.sort(eigvals, descending=self.descending)
                solver_state.cache["eigvecs"] = eigvecs[:, idcs]

            # Return approximate eigenvectors according to strategy
            if solver_state.iteration < solver_state.cache["eigvecs"].shape[1]:
                return solver_state.cache["eigvecs"][:, solver_state.iteration]
            else:
                return solver_state.residual
