#!/usr/bin/env python3

from __future__ import annotations

import torch

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
        base_action = self.base_policy(solver_state)
        mixin_action = self.mixin_policy(solver_state)

        action = (1.0 - self.mixin_coeff) * base_action + self.mixin_coeff * mixin_action
        action.requires_grad_(self._mixin_coeff_logit_transformed.requires_grad)

        return action

    @property
    def mixin_coeff(self) -> torch.Tensor:
        return torch.sigmoid(self._mixin_coeff_logit_transformed)
