#!/usr/bin/env python3

from __future__ import annotations

from typing import Iterable

import torch

from .linear_solver_policy import LinearSolverPolicy


class SwitchingPolicy(LinearSolverPolicy):
    """Policy choosing actions from a set of policies in sequence.

    :param policies: Iterable of policies to use.
    :param switching_points: Iterable of points at which to switch to the
    next policy. Assumed to be sorted.
    """

    def __init__(
        self,
        policies: Iterable[LinearSolverPolicy],
        switching_points: torch.Tensor,
    ) -> None:
        super().__init__()
        self.policies = policies
        self.switching_points = torch.as_tensor(switching_points)

        if len(self.switching_points) != (len(self.policies) - 1):
            raise ValueError(
                f"Number of switching points ({len(self.switching_points)})"
                + " not compatible with number of policies ({len(self.policies)})."
            )

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        idx_current_policy = torch.searchsorted(self.switching_points, solver_state.iteration, side="right").item()
        return self.policies[idx_current_policy](solver_state)
