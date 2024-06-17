#!/usr/bin/env python3
from abc import ABC, abstractmethod

import torch
from torch import nn

from ..linear_solver import LinearSolverState


class LinearSolverPolicy(ABC, nn.Module):
    """Policy of a linear solver.

    A linear solver policy chooses actions to observe the linear system :math:`Ax_* = b`.
    """

    @abstractmethod
    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        """Generate an action.

        :param solver_state:
        """
        raise NotImplementedError
