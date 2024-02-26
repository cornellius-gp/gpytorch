"""Preconditioner to be used as part of an augmented prior."""

import abc
from typing import Optional

import torch
from torch import nn


class Preconditioner(nn.Module, abc.ABC):
    # TODO: do we want this to inherit from linear operator?
    """Abstract base class for a kernel matrix preconditioner."""

    def __init__(self, kernel, noise, X, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel
        self.noise = noise
        self.X = X

    def forward(self, input):
        raise NotImplementedError

    @property
    def shape(self):
        return (self.X.shape[0], self.X.shape[0])

    def inv_matmul(self, input):
        raise NotImplementedError

    def sqrt_inv_matmul(self, input):
        raise NotImplementedError

    def _scaling(self, upper_bound_max_eigval_preconditioner_inv: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure preconditioner validity
        if upper_bound_max_eigval_preconditioner_inv is None:
            upper_bound_max_eigval_preconditioner_inv = torch.max(
                torch.sum(torch.abs(self.inv_matmul(torch.eye(self.X.shape[0]))), dim=1)
            )
        if torch.abs(upper_bound_max_eigval_preconditioner_inv) < 1e-12:
            return torch.zeros(())

        upper_bound_max_eigval_Khat = torch.sum(self.kernel(self.X)) / self.X.shape[0] + self.noise
        lower_bound_max_eigval_Khat = torch.max(
            torch.sum(
                torch.abs(self.kernel(self.X).to_dense() + self.noise * torch.eye(self.X.shape[0])),
                dim=1,
            )
        )
        scalar_factor_precond_inv = (
            torch.minimum(lower_bound_max_eigval_Khat, 1.0 / upper_bound_max_eigval_Khat)
            / upper_bound_max_eigval_preconditioner_inv
        )
        # print(scalar_factor_precond_inv)
        return scalar_factor_precond_inv


# TODO: need a caching mechanism so we only recompute the preconditioner when the hyperparameters are updated
