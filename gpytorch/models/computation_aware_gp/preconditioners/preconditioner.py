"""Preconditioner to be used as part of an augmented prior."""

import abc
from typing import Optional

import torch
from torch import nn


class Preconditioner(nn.Module, abc.ABC):
    # TODO: do we want this to inherit from linear operator?
    """Abstract base class for a kernel matrix preconditioner."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(
        self,
        input: torch.Tensor,
        kernel,
        noise: torch.Tensor,
        X: torch.Tensor,
    ):
        raise NotImplementedError

    def inv_matmul(
        self,
        input: torch.Tensor,
        kernel,
        noise: torch.Tensor,
        X: torch.Tensor,
    ):
        raise NotImplementedError

    def sqrt_inv_matmul(
        self,
        input: torch.Tensor,
        kernel,
        noise: torch.Tensor,
        X: torch.Tensor,
    ):
        raise NotImplementedError

    def _scaling(
        self,
        kernel,
        noise: torch.Tensor,
        X: torch.Tensor,
        unnormalized_preconditioner_inv: Optional[torch.Tensor] = None,
        upper_bound_max_eigval_preconditioner_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure preconditioner validity
        # if upper_bound_max_eigval_preconditioner_inv is None:
        #     upper_bound_max_eigval_preconditioner_inv = torch.max(
        #         torch.sum(torch.abs(unnormalized_preconditioner_inv), dim=1)
        #     )
        # if torch.abs(upper_bound_max_eigval_preconditioner_inv) < 1e-12:
        #     return torch.zeros(())

        # lower_bound_max_eigval_Khat = torch.sum(kernel(X)) / X.shape[0] + noise
        # upper_bound_max_eigval_Khat = torch.max(
        #     torch.sum(
        #         torch.abs(kernel(X).to_dense() + noise * torch.eye(X.shape[0])),
        #         dim=1,
        #     )
        # )
        # scalar_factor_precond_inv = (
        #     torch.minimum(lower_bound_max_eigval_Khat, 1.0 / upper_bound_max_eigval_Khat)
        #     / upper_bound_max_eigval_preconditioner_inv
        # )
        Pinvsqrt_Khat = self.sqrt_inv_matmul(
            kernel(X).to_dense() + noise * torch.eye(X.shape[0]), kernel=kernel, noise=noise, X=X
        )
        Pinvsqrt_Khat_Pinvsqrt = self.sqrt_inv_matmul(Pinvsqrt_Khat.mT, kernel=kernel, noise=noise, X=X)
        upper_bound_max_eigval_Pinvsqrt_Khat_Pinvsqrt = torch.max(torch.sum(torch.abs(Pinvsqrt_Khat_Pinvsqrt), dim=1))
        scalar_factor_precond_inv = 1 / upper_bound_max_eigval_Pinvsqrt_Khat_Pinvsqrt

        # print(scalar_factor_precond_inv)
        return scalar_factor_precond_inv


# TODO: need a caching mechanism so we only recompute the preconditioner when the hyperparameters are updated
