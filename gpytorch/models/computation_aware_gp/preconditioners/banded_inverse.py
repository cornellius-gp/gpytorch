"""Banded """

from typing import Optional

import torch
from torch import nn

from .preconditioner import Preconditioner


class Diagonal(Preconditioner):
    """Diagonal preconditioner."""

    def __init__(
        self, kernel, noise, X, inverse_diagonal_sqrt_unnormalized: Optional[torch.Tensor] = None, **kwargs
    ) -> None:
        super().__init__(kernel, noise, X, **kwargs)
        if inverse_diagonal_sqrt_unnormalized is None:
            self.inverse_diagonal_sqrt_unnormalized = nn.Parameter(torch.ones((self.X.shape[0],)))
        else:
            self.inverse_diagonal_sqrt_unnormalized = nn.Parameter(inverse_diagonal_sqrt_unnormalized)

        self.inverse_diagonal_sqrt = self.inverse_diagonal_sqrt_unnormalized * torch.sqrt(
            self._scaling(torch.max(self.inverse_diagonal_sqrt_unnormalized**2))
        )

    def inv_matmul(self, input):
        if input.ndim <= 1:
            return self.inverse_diagonal_sqrt**2 * input
        return (self.inverse_diagonal_sqrt**2).reshape(-1, 1) * input

    def sqrt_inv_matmul(self, input):
        if input.ndim <= 1:
            return self.inverse_diagonal_sqrt * input
        return self.inverse_diagonal_sqrt.reshape(-1, 1) * input

    # def forward(self, input):
    #     return self.sqrt_inv_matmul(input) # TODO: causes same bug


class BandedInverse(Preconditioner):
    """Preconditioner with banded inverse."""

    pass


class ToeplitzInverse(Preconditioner):
    """Preconditioner with (banded) Toeplitz inverse."""

    pass
