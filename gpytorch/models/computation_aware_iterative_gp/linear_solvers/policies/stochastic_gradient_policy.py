#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch
from linear_operator.operators import BlockDiagonalSparseLinearOperator, LinearOperator

# from ..... import kernels

from .linear_solver_policy import LinearSolverPolicy


class StochasticGradientPolicy(LinearSolverPolicy):
    """Policy choosing stochastic gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions."""

    def __init__(
        self, batch_size: int = 64, preconditioner: Optional[LinearOperator] = None, dense: bool = False
    ) -> None:
        self.num_nonzero = batch_size
        self.preconditioner = preconditioner
        self.dense = dense

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            residual = solver_state.residual

            # Sample non-zero indices (i.e. batches to compute gradients for)
            perm = torch.randperm(solver_state.problem.A.shape[0])
            non_zero_idcs = perm[0 : self.num_nonzero]

            if residual is not None:
                if isinstance(self.preconditioner, (torch.Tensor, LinearOperator)):
                    residual = self.preconditioner @ residual
                elif callable(self.preconditioner):
                    residual = self.preconditioner(residual).squeeze()
                residual_batch = residual[non_zero_idcs]
            else:
                # outputscale = 1.0
                # lengthscale = 1.0
                # kernel = solver_state.cache["kernel"]
                # if isinstance(kernel, kernels.ScaleKernel):
                #     outputscale = kernel.outputscale
                #     lengthscale = kernel.base_kernel.lengthscale
                #     forward_fn = kernel.base_kernel._forward
                # else:
                #     try:
                #         lengthscale = kernel.lengthscale
                #     except AttributeError:
                #         pass

                # forward_fn = kernel._forward
                # # Use sparse bilinear form here to get stochastic gradient
                # SKS, SS = kernels.SparseBilinearForms.apply(
                #     solver_state.cache["train_inputs"] / lengthscale,
                #     solver_state.cache["actions_op"].blocks.mT,
                #     action.blocks.mT,
                #     solver_state.cache["actions_op"].non_zero_idcs.mT,
                #     action.non_zero_idcs.mT,
                #     forward_fn,
                #     None,  # vjp_fn,
                #     None,
                # )
                # actions_linear_op_current_action = (outputscale * SKS + noise * SS).reshape((-1,))

                # TODO: simply evaluate the kernel on subset data and then multiply with z_i:
                # k(X[non_zero_idcs, :], X[action.non_zero_idcs, :]) + sigma2 * non_zero_idcs == action.non_zero_idcs
                raise NotImplementedError

            if self.dense:
                # Stochastic gradient (residual)
                action = torch.zeros(
                    residual.shape[0],
                    dtype=residual.dtype,
                    device=residual.device,
                )
                action[non_zero_idcs] = residual_batch
                return action

            else:
                return BlockDiagonalSparseLinearOperator(
                    non_zero_idcs=non_zero_idcs, blocks=residual_batch, size_sparse_dim=len(perm)
                )
