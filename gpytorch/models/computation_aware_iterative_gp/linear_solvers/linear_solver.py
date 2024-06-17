#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from linear_operator.operators import LinearOperator

from torch import Tensor

__all__ = ["LinearSolver", "LinearSolverState"]


@dataclass
class LinearSystem:
    """Linear system :math:`Ax_* = b`."""

    A: LinearOperator
    b: LinearOperator


@dataclass
class LinearSolverState:
    """State of a linear solver applied to :math:`Ax_*=b`.

    :param problem: Linear system to solve.
    :param solution: Approximate solution of the linear system :math:`x_i \\approx A^{-1}b`.
    :param forward_op: Estimate of the forward operation :math:`A`.
    :param inverse_op: Estimate of the inverse operation :math:`A^{-1}`.
    :param residual: Residual :math:`r_i = b - Ax_i`.
    :param residual_norm: Residual norm :math:`\\lVert r_i \\rVert_2`.
    :param logdet: Estimate of the log-determinant :math:`\\log \\operatorname{det}(A)`.
    :param iteration: Iteration of the solver.
    :param cache: Miscellaneous quantities cached by the solver.
    """

    problem: LinearSystem
    solution: Tensor
    forward_op: LinearOperator
    inverse_op: LinearOperator
    residual: Tensor
    residual_norm: Tensor
    logdet: Tensor  # TODO: move into cache
    iteration: int
    cache: dict = field(default_factory=dict)


class LinearSolver(ABC):
    r"""Abstract base class for linear solvers.

    Method which solves a linear system of the form

    .. math:: Ax_*=b.
    """

    @abstractmethod
    def solve(self, linear_op: LinearOperator, rhs: Tensor, /, **kwargs) -> LinearSolverState:
        r"""Solve linear system :math:`Ax_*=b`.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """
        raise NotImplementedError
