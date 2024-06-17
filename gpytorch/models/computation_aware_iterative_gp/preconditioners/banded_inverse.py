"""Banded """

from typing import Optional

import torch
from torch import nn

from .preconditioner import Preconditioner


class Diagonal(Preconditioner):
    """Diagonal preconditioner."""

    def __init__(self, inverse_diagonal_sqrt_unnormalized: torch.Tensor = torch.ones(()), **kwargs) -> None:
        super().__init__(**kwargs)
        self.inverse_diagonal_sqrt_unnormalized = nn.Parameter(inverse_diagonal_sqrt_unnormalized)

    def _inverse_diagonal_sqrt(self, kernel, noise: torch.Tensor, X: torch.Tensor):
        scaling_factor = self._scaling(
            kernel=kernel,
            noise=noise,
            X=X,
            upper_bound_max_eigval_preconditioner_inv=torch.max(self.inverse_diagonal_sqrt_unnormalized**2),
        )
        return self.inverse_diagonal_sqrt_unnormalized * torch.sqrt(scaling_factor)
        # TODO: need caching mechanism for efficiency here!

    def inv_matmul(
        self,
        input: torch.Tensor,
        kernel,
        noise: torch.Tensor,
        X: torch.Tensor,
    ):
        inverse_diagonal_sqrt = self._inverse_diagonal_sqrt(kernel=kernel, noise=noise, X=X)

        if input.ndim <= 1:
            return inverse_diagonal_sqrt**2 * input
        return (inverse_diagonal_sqrt**2).reshape(-1, 1) * input

    def sqrt_inv_matmul(
        self,
        input: torch.Tensor,
        kernel,
        noise: torch.Tensor,
        X: torch.Tensor,
    ):
        inverse_diagonal_sqrt = self._inverse_diagonal_sqrt(kernel=kernel, noise=noise, X=X)

        if input.ndim <= 1:
            return inverse_diagonal_sqrt * input
        return inverse_diagonal_sqrt.reshape(-1, 1) * input

    # def forward(
    #     self,
    #     input: torch.Tensor,
    #     kernel,
    #     noise: torch.Tensor,
    #     X: torch.Tensor,
    # ):
    #     return self.sqrt_inv_matmul(input)


class BandedInverse(Preconditioner):
    """Preconditioner with banded inverse."""

    # TODO: inverse has bands with O(n_bands n_data) learnable parameters
    pass


class ToeplitzInverse(Preconditioner):
    """Preconditioner with (banded) Toeplitz inverse."""

    pass

    # def __init__(self, values_bands: torch.Tensor = torch.ones((3,)), **kwargs) -> None:
    #     super().__init__(**kwargs)
    #     self.values_bands = nn.Parameter(values_bands)

    # @staticmethod
    # def naive_toeplitz(values_bands: torch.Tensor):
    #     values_bands = torch.cat((values_bands, values_bands[1:].flip(0)))
    #     i, j = torch.ones(len(values_bands), len(values_bands)).nonzero().T
    #     return values_bands[j - i].reshape(len(values_bands), len(values_bands))

    # def inv_matmul(self, input: torch.Tensor, kernel, noise: torch.Tensor, X: torch.Tensor):
    #     toeplitz_preconditioner_inv = ToeplitzInverse.naive_toeplitz(
    #         values_bands=torch.concat([self.values_bands, torch.zeros((X.shape[0] - len(self.values_bands),))])
    #     )

    #     scaling_factor = self._scaling(
    #         kernel=kernel,
    #         noise=noise,
    #         X=X,
    #         upper_bound_max_eigval_preconditioner_inv=None,  # TODO: we can give an upper bound here for Toeplitz matrices
    #     )

    #     return scaling_factor * toeplitz_preconditioner_inv @ input
