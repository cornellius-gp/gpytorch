#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Union

import torch
from linear_operator import operators

from gpytorch import settings

from gpytorch.kernels import Kernel
from ._sparsify_vector import sparsify_vector
from .linear_solver_policy import LinearSolverPolicy


class PseudoInputPolicy(LinearSolverPolicy):
    """Policy choosing kernel functions evaluated at (pseudo-) inputs as actions.

    .. math :: s_i = k(X, z_i)

    :param precond: Preconditioner :math:`P \\approx A^{-1}`.
    """

    def __init__(
        self,
        kernel: Kernel,
        train_data: torch.Tensor,
        pseudo_inputs: torch.Tensor,
        sparsification_threshold: float = 0.0,
        num_non_zero: Optional[int] = None,
        optimize_pseudo_inputs: bool = True,
        kernel_hyperparams_as_policy_hyperparams: bool = False,
        precondition: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.kernel.to(device=train_data.device, dtype=train_data.dtype)
        self.train_data = train_data

        # Register kernel hyperparameters as policy parameters
        if kernel_hyperparams_as_policy_hyperparams:
            for kernel_hyperparam_name, kernel_hyperparam in self.kernel.named_parameters():
                self.register_parameter(kernel_hyperparam_name, kernel_hyperparam)

        if optimize_pseudo_inputs:
            self.pseudo_inputs = torch.nn.Parameter(pseudo_inputs.clone())
        else:
            self.pseudo_inputs = pseudo_inputs.clone()

        self.sparsification_threshold = sparsification_threshold
        self.num_nonzero = num_non_zero
        self.precondition = precondition

        # Compute preconditioner
        if self.precondition:

            def partial_cholesky_preconditioner(kernel, rank=100):
                with settings.min_preconditioning_size(rank), settings.max_preconditioner_size(rank):
                    return (
                        kernel(train_data) + 0.01 * operators.IdentityLinearOperator(len(train_data))
                    )._solve_preconditioner()

            K_XZ = self.kernel(train_data, self.pseudo_inputs).to_dense()
            # self.preconditioner = torch.linalg.cholesky(
            #     K_XZ.mT @ K_XZ + 10 * torch.eye(len(self.train_data)),
            #     upper=False,  # 0.002 * self.kernel(self.pseudo_inputs).to_dense(), upper=False
            # )
            # self.actions = torch.linalg.solve_triangular(self.preconditioner, K_XZ.mT, upper=False).mT
            # self.actions = torch.cholesky_solve(K_XZ.mT, self.preconditioner, upper=False).mT
            self.actions = partial_cholesky_preconditioner(self.kernel)(K_XZ)

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        action = (
            self.kernel(
                self.train_data,
                self.pseudo_inputs[solver_state.iteration].reshape(1, -1),
            )
            .evaluate_kernel()
            .to_dense()
        ).reshape(-1)

        if self.precondition:
            action = self.actions[:, solver_state.iteration].reshape(-1)

        if self.sparsification_threshold > 0.0:
            action[torch.abs(action) < self.sparsification_threshold] = 0.0

        # Sparsify
        if self.num_nonzero is not None:
            action = sparsify_vector(action, num_non_zero=self.num_nonzero)

        return action

    def __setattr__(self, name: str, value: Union[torch.Tensor, torch.nn.Module]) -> None:
        """Overwrite __set_attr__ to not register kernel hyperparameters as hyperparameters of a policy."""
        if not (name.startswith("kernel") or name.startswith("likelihood") or name.startswith("covar_module")):
            super().__setattr__(name=name, value=value)
        else:
            object.__setattr__(self, name, value)

    # def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
    #     if not (name.startswith("likelihood") or name.startswith("covar_module")):
    #         super().register_parameter(name=name, param=param)
