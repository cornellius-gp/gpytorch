#!/usr/bin/env python3

from __future__ import annotations

from typing import Generator, Optional

import torch

from linear_operator import settings
from linear_operator.operators import BlockSparseLinearOperator, LinearOperator, to_linear_operator, ZeroLinearOperator
from torch import Tensor

from .linear_solver import LinearSolver, LinearSolverState, LinearSystem


class ProbabilisticLinearSolver(LinearSolver):
    """Probabilistic linear solver.

    Iteratively solve linear systems of the form

    .. math:: Ax_* = b

    where :math:`A` is a (symmetric positive-definite) linear operator. A probabilistic
    linear solver chooses actions :math:`s_i` in each iteration to observe the residual
    by computing :math:`\\alpha_i = s_i^\\top (b - Ax_i)`.

    :param policy: Policy selecting actions :math:`s_i` to probe the residual with.
    :param abstol: Absolute residual tolerance.
    :param reltol: Relative residual tolerance.
    :max_iter: Maximum number of iterations. Defaults to `10 * rhs.shape[0]`.
    """

    def __init__(
        self,
        policy: "LinearSolverPolicy",
        abstol: float = 1e-5,
        reltol: float = 1e-5,
        max_iter: int = None,
    ):
        self.policy = policy
        self.abstol = abstol
        self.reltol = reltol
        self.max_iter = max_iter
        self.enable_action_gradients = True if len(list(self.policy.parameters())) > 0 else False

    def solve_iterator(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> Generator[LinearSolverState, None, None]:
        r"""Generator implementing the linear solver iteration.

        This function allows stepping through the solver iteration one step at a time and thus exposes internal
        quantities in the solver state cache.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """
        # Setup
        linear_op = to_linear_operator(linear_op)
        if self.max_iter is None:
            max_iter = 10 * rhs.shape[0]
        else:
            max_iter = self.max_iter

        # Ensure initial guess and rhs are vectors
        rhs = rhs.reshape(-1)
        if x is not None:
            x = x.reshape(-1)

        if x is None:
            x = torch.zeros_like(rhs, requires_grad=True)
            inverse_op = ZeroLinearOperator(*linear_op.shape, dtype=linear_op.dtype, device=linear_op.device)
            residual = rhs
            logdet = torch.zeros((), requires_grad=True)
        else:
            raise NotImplementedError("Currently we do not support initializing with a given solution x.")

        # Initialize solver state
        solver_state = LinearSolverState(
            problem=LinearSystem(A=linear_op, b=rhs),
            solution=x,
            forward_op=None,
            inverse_op=inverse_op,
            residual=residual,
            residual_norm=torch.linalg.vector_norm(residual, ord=2),
            logdet=logdet,
            iteration=0,
            cache={
                "schur_complements": [],
                "rhs_norm": torch.linalg.vector_norm(rhs, ord=2),
                "action": None,
                "actions_op": None,
                # "linear_op_actions": None,
                "observation": None,
                "compressed_solution": None,
            },
        )

        yield solver_state

        while True:
            # Check convergence
            if (
                solver_state.residual_norm < max(self.abstol, self.reltol * solver_state.cache["rhs_norm"])
                or solver_state.iteration >= max_iter
            ):
                break

            with torch.set_grad_enabled(
                self.enable_action_gradients
            ):  # For efficiency, only track gradients if necessary to optimize action hyperparameters.
                # Select action
                action = self.policy(solver_state)

            with torch.no_grad():  # Saves 2x compute since we don't need gradients through the solve.
                linear_op_action = (
                    (action._matmul(linear_op)) if isinstance(action, BlockSparseLinearOperator) else linear_op @ action
                ).squeeze()

                if solver_state.cache["actions_op"] is not None:
                    # TODO: operation is not yet sparse, so it costs O(ni) per iteration => O(ni^2) unnecessary cost
                    prev_actions_linear_op_action = solver_state.cache["actions_op"] @ linear_op_action
                else:
                    prev_actions_linear_op_action = None

                # Observation
                observ = action @ solver_state.residual

                # Schur complement / Squared linop-norm of search direction
                action_linear_op_action = action @ linear_op_action

                schur_complement = action_linear_op_action

                if solver_state.cache["actions_op"] is not None:
                    gram_inv_tilde_z = torch.cholesky_solve(
                        prev_actions_linear_op_action.reshape(-1, 1),
                        solver_state.cache["cholfac_gram"],
                        upper=False,
                    ).reshape(-1)

                    schur_complement = schur_complement - torch.inner(prev_actions_linear_op_action, gram_inv_tilde_z)

                solver_state.cache["schur_complements"].append(schur_complement)

                if schur_complement <= 0:
                    if settings.verbose_linalg.on():
                        settings.verbose_linalg.logger.debug(
                            f"PLS terminated after {solver_state.iteration} iteration(s)"
                            + " due to a negative Schur complement."
                        )
                    break

            if solver_state.cache["actions_op"] is None:
                # Matrix of previous actions
                solver_state.cache["actions_op"] = (
                    action if isinstance(action, BlockSparseLinearOperator) else torch.reshape(action, (1, -1))
                )

                with torch.no_grad():
                    # # Matrix of previous actions applied to the kernel matrix
                    # solver_state.cache["linear_op_actions"] = torch.reshape(linear_op_action, (-1, 1))

                    # Initialize Cholesky factor
                    solver_state.cache["cholfac_gram"] = torch.sqrt(action_linear_op_action).reshape(1, 1)

            else:
                with torch.no_grad():
                    # Update to Cholesky factor of Gram matrix S_i'\hat{K}S_i
                    new_cholfac_bottom_row_minus_last_entry = torch.linalg.solve_triangular(
                        solver_state.cache["cholfac_gram"],
                        prev_actions_linear_op_action.reshape(-1, 1),
                        upper=False,
                    ).reshape(-1)
                    new_cholfac_bottom_row_rightmost_entry = torch.sqrt(schur_complement)

                    solver_state.cache["cholfac_gram"] = torch.vstack(
                        (
                            torch.hstack(
                                (
                                    solver_state.cache["cholfac_gram"],
                                    torch.zeros(
                                        (solver_state.iteration, 1),
                                        device=linear_op.device,
                                        dtype=linear_op.dtype,
                                    ),
                                )
                            ),
                            torch.hstack(
                                (
                                    new_cholfac_bottom_row_minus_last_entry,
                                    new_cholfac_bottom_row_rightmost_entry,
                                )
                            ),
                        )
                    )

                # Matrix of actions
                if isinstance(action, BlockSparseLinearOperator):
                    solver_state.cache["actions_op"] = BlockSparseLinearOperator(
                        non_zero_idcs=torch.cat(
                            (solver_state.cache["actions_op"].non_zero_idcs, action.non_zero_idcs), dim=0
                        ),
                        blocks=torch.cat((solver_state.cache["actions_op"].blocks, action.blocks), dim=0),
                        size_sparse_dim=solver_state.problem.A.shape[0],
                    )
                else:
                    solver_state.cache["actions_op"] = torch.vstack(
                        (solver_state.cache["actions_op"], action.reshape(1, -1))
                    )

                # with torch.no_grad():
                #     # Matrix of actions applied to the kernel matrix
                #     solver_state.cache["linear_op_actions"] = torch.hstack(
                #         (
                #             solver_state.cache["linear_op_actions"],
                #             linear_op_action.reshape(-1, 1),
                #         )
                #     )

            with torch.no_grad():
                # Update compressed solution estimate
                solver_state.cache["compressed_solution"] = torch.cholesky_solve(
                    # TODO: operation is not yet sparse, so it costs O(ni) per iteration => O(ni^2) unnecessary cost
                    (solver_state.cache["actions_op"] @ rhs).reshape(-1, 1),
                    solver_state.cache["cholfac_gram"],
                    upper=False,
                ).reshape(-1)

                # Update solution estimate
                # solver_state.solution = solver_state.cache["actions_op"] @ solver_state.cache["compressed_solution"]

                # Update residual
                solver_state.residual = (
                    solver_state.residual - linear_op_action * solver_state.cache["compressed_solution"][-1]
                )

                # # Compute residual
                # solver_state.cache["linear_op_actions_compressed_solution"] = (
                #     solver_state.cache["linear_op_actions"] @ solver_state.cache["compressed_solution"]
                # )
                # solver_state.residual = (
                #     solver_state.problem.b - solver_state.cache["linear_op_actions_compressed_solution"]
                # )
                # # Explicitly recomputing the residual improves stability a bit (for CG)
                #
                # solver_state.residual = (
                #     solver_state.problem.b - solver_state.problem.A @ solver_state.solution
                # )

                solver_state.residual_norm = torch.linalg.vector_norm(solver_state.residual, ord=2)
                # TODO: should we check for an increase in residual here to stop early?

                # Update inverse approximation
                solver_state.inverse_op = None  # TODO: lazy representation for simpler code?

                # Update log-determinant
                solver_state.logdet = solver_state.logdet + torch.log(schur_complement)

                # Update iteration
                solver_state.iteration += 1

                # Update solver state cache
                solver_state.cache["action"] = action
                solver_state.cache["observation"] = observ

                yield solver_state

    def solve(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> LinearSolverState:
        r"""Solve linear system :math:`Ax_*=b`.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """

        solver_state = None

        for solver_state in self.solve_iterator(linear_op, rhs, x=x):
            pass

        return solver_state
