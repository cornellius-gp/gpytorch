from typing import List

import numpy as np

import torch
from torch import nn

from .preconditioner import Preconditioner


class SparseInverseCholesky(Preconditioner):
    """Naive implementation of a sparse precision matrix / Vecchia approximation.

    Implementation based on Theorem 2.1 of Schaefer et al. "Sparse Cholesky factorization by Kullback-Leibler minimization" 2021.
    """

    def __init__(self, num_diags_cholfac: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_diags_cholfac = num_diags_cholfac

    def _cholfac_precision(self, kernel, noise: torch.Tensor, X: torch.Tensor):
        cholfac_precision = torch.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            sparsity_set = X[
                i : np.minimum(i + self.num_diags_cholfac, X.shape[0])
            ]  # NOTE: works particularly well if training data is ordered

            cholfac_K_sparsity_set = torch.linalg.cholesky(
                kernel(sparsity_set).evaluate() + noise * torch.eye(len(sparsity_set)),
                upper=False,
            )

            with torch.no_grad():
                unit_vec = torch.zeros((len(sparsity_set), 1))
                unit_vec[0, 0] = 1.0

            cholfac_K_neighbor_set_unit_vector = torch.cholesky_solve(unit_vec, cholfac_K_sparsity_set, upper=False)
            cholfac_precision[i : np.minimum(i + self.num_diags_cholfac, X.shape[0]), i] = (
                cholfac_K_neighbor_set_unit_vector
                / torch.sqrt(unit_vec.mT @ cholfac_K_neighbor_set_unit_vector).squeeze()
            ).reshape(
                -1
            )  # Note: Cholesky factor has sparsity pattern of choice! Here: banded and lower triangular -> efficient!

        return cholfac_precision

    def sqrt_inv_matmul(self, input: torch.Tensor, kernel, noise: torch.Tensor, X: torch.Tensor):

        cholfac_precision = self._cholfac_precision(kernel=kernel, noise=noise, X=X)
        return cholfac_precision.mT @ input

    def inv_matmul(self, input: torch.Tensor, kernel, noise: torch.Tensor, X: torch.Tensor):

        cholfac_precision = self._cholfac_precision(kernel=kernel, noise=noise, X=X)
        banded_inverse = cholfac_precision @ cholfac_precision.mT

        # Normalize
        scaling_factor = self._scaling(
            kernel=kernel,
            noise=noise,
            X=X,
            unnormalized_preconditioner_inv=banded_inverse,
            upper_bound_max_eigval_preconditioner_inv=None,
        )  # TODO: can we find a tighter upper bound for banded matrices?

        scaling_factor = 1.0 / 8.0  # TODO: REMOVE ME

        return (scaling_factor * banded_inverse) @ input


class BandedInverseCholesky(Preconditioner):
    def __init__(self, unnormalized_bands: List[torch.Tensor], **kwargs) -> None:
        super().__init__(**kwargs)
        self.unnormalized_bands = nn.ParameterList(unnormalized_bands)

    def sqrt_inv_matmul(self, input: torch.Tensor, kernel, noise: torch.Tensor, X: torch.Tensor):
        raise NotImplementedError

    def inv_matmul(self, input: torch.Tensor, kernel, noise: torch.Tensor, X: torch.Tensor):
        # Naive implementation
        # TODO: use banded Cholesky structure for efficiency!

        # Banded Cholesky factor
        cholfac = torch.zeros(size=(X.shape[0], X.shape[0]))
        for i in range(len(self.unnormalized_bands)):
            cholfac = cholfac + torch.diag(self.unnormalized_bands[i], diagonal=-i)

        banded_inverse = cholfac @ cholfac.mT

        # Normalize
        scaling_factor = self._scaling(
            kernel=kernel,
            noise=noise,
            X=X,
            unnormalized_preconditioner_inv=banded_inverse,
            upper_bound_max_eigval_preconditioner_inv=None,
        )  # TODO: can we find a tighter upper bound for banded matrices?

        return (scaling_factor * banded_inverse) @ input
