#!/usr/bin/env python3

from .linear_solver import LinearSolver, LinearSolverState
from .probabilistic_linear_solver import ProbabilisticLinearSolver

# Convenience alias
PLS = ProbabilisticLinearSolver

__all__ = [
    "LinearSolver",
    "LinearSolverState",
    "ProbabilisticLinearSolver",
]
