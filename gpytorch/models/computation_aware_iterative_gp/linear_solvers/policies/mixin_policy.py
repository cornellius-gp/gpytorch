#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator.operators import BlockDiagonalSparseLinearOperator

from .linear_solver_policy import LinearSolverPolicy


class MixinPolicy(LinearSolverPolicy):
    """Policy choosing actions as a linear combination of actions from two policies."""

    def __init__(
        self,
        base_policy: LinearSolverPolicy,
        mixin_policy: LinearSolverPolicy,
        mixin_coeff: float,
        optimize_mixin_coeff: bool = True,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.mixin_policy = mixin_policy

        # TODO: This exists to avoid vanishing gradients. Is there a better way to do this?
        eps = 1e-6
        if mixin_coeff == 0.0:
            mixin_coeff += eps
        if mixin_coeff == 1.0:
            mixin_coeff -= eps

        self._mixin_coeff_logit_transformed = torch.nn.Parameter(torch.logit(torch.as_tensor(mixin_coeff)))
        self._mixin_coeff_logit_transformed.requires_grad = optimize_mixin_coeff

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        # Generate base action from policy
        base_action = self.base_policy(solver_state)

        # Generate mixin action (potentially with matched sparsity)
        mixin_action = self.mixin_policy(solver_state)

        if isinstance(base_action, torch.Tensor) and isinstance(mixin_action, torch.Tensor):
            # Compute resulting convex combination of actions
            action = (1.0 - self.mixin_coeff) * base_action + self.mixin_coeff * mixin_action
        elif isinstance(base_action, BlockDiagonalSparseLinearOperator) and isinstance(
            mixin_action, BlockDiagonalSparseLinearOperator
        ):
            # Create union of non-zero index tensors
            intersection_mask = base_action.non_zero_idcs.view(1, -1) == mixin_action.non_zero_idcs.view(-1, 1)
            intersection_mask_base_action = (intersection_mask).any(dim=0)
            intersection_mask_mixin_action = (intersection_mask).any(dim=1)

            non_zero_idcs = torch.cat(
                (
                    base_action.non_zero_idcs[..., ~intersection_mask_base_action],
                    mixin_action.non_zero_idcs[..., ~intersection_mask_mixin_action],
                    base_action.non_zero_idcs[..., intersection_mask_base_action],
                ),
                dim=1,
            )
            blocks = torch.cat(
                (
                    (1.0 - self.mixin_coeff) * base_action.blocks[..., ~intersection_mask_base_action],
                    self.mixin_coeff * mixin_action.blocks[..., ~intersection_mask_mixin_action],
                    (1.0 - self.mixin_coeff) * base_action.blocks[..., intersection_mask_base_action]
                    + self.mixin_coeff * mixin_action.blocks[..., intersection_mask_mixin_action],
                ),
                dim=1,
            )
            blocks.requires_grad_(self._mixin_coeff_logit_transformed.requires_grad)

            action = BlockDiagonalSparseLinearOperator(
                non_zero_idcs=non_zero_idcs, blocks=blocks, size_sparse_dim=base_action.size_sparse_dim
            )
        else:
            raise NotImplementedError

        action.requires_grad_(self._mixin_coeff_logit_transformed.requires_grad)
        return action

    @property
    def mixin_coeff(self) -> torch.Tensor:
        return torch.sigmoid(self._mixin_coeff_logit_transformed)
