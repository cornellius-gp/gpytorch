"""Preconditioners for kernel matrices."""

import numpy as np
import torch
from linear_operator.operators import DiagLinearOperator

from .... import settings
from .banded_inverse import BandedInverse, Diagonal, ToeplitzInverse

from .preconditioner import Preconditioner

from .scalar import Scalar
from .sparse_inverse_cholesky import BandedInverseCholesky, SparseInverseCholesky

Vecchia = SparseInverseCholesky

__all__ = [
    "Preconditioner",
    "Scalar",
    "Diagonal",
    "Vecchia",
    "SparseInverseCholesky",
    "BandedInverseCholesky",
    "BandedInverse",
    "ToeplitzInverse",
]


def scalar(
    kernel,
    noise,
    X_train,
):
    lmda_max_upper_bound = torch.max(torch.sum(kernel(X_train), dim=1))
    return DiagLinearOperator(1 / (lmda_max_upper_bound + noise) * torch.ones(X_train.shape[0]))


def partial_cholesky(
    kernel,
    noise,
    X_train,
    precond_size,
    rescale: bool = True,
):
    with settings.min_preconditioning_size(precond_size), settings.max_preconditioner_size(precond_size):
        preconditioner_inv = (
            kernel(X_train) + DiagLinearOperator(noise * torch.ones(X_train.shape[0]))
        )._solve_preconditioner()(torch.eye(X_train.shape[0]))
    if rescale:
        preconditioner_inv = rescale_preconditioner(
            preconditioner_inv=preconditioner_inv, kernel=kernel, noise=noise, X_train=X_train
        )
    return preconditioner_inv


def sparse_precision(
    kernel,
    noise,
    X_train,
    num_diags_cholfac=1,
    rescale: bool = True,
):
    """Naive implementation of a sparse precision matrix / Vecchia approximation.

    Implementation based on Theorem 2.1 of Schaefer et al. "Sparse Cholesky factorization by Kullback-Leibler minimization" 2021.
    """

    cholfac_precision = torch.zeros((X_train.shape[0], X_train.shape[0]))

    for i in range(X_train.shape[0]):
        sparsity_set = X_train[
            i : np.minimum(i + num_diags_cholfac, X_train.shape[0])
        ]  # NOTE: works particularly well if training data is ordered

        cholfac_K_sparsity_set = torch.linalg.cholesky(
            kernel(sparsity_set).evaluate() + noise * torch.eye(len(sparsity_set)),
            upper=False,
        )

        with torch.no_grad():
            unit_vec = torch.zeros((len(sparsity_set), 1))
            unit_vec[0, 0] = 1.0

        cholfac_K_neighbor_set_unit_vector = torch.cholesky_solve(unit_vec, cholfac_K_sparsity_set, upper=False)
        cholfac_precision[i : np.minimum(i + num_diags_cholfac, X_train.shape[0]), i] = (
            cholfac_K_neighbor_set_unit_vector / (unit_vec.mT @ cholfac_K_neighbor_set_unit_vector).squeeze()
        ).reshape(
            -1
        )  # Note: Cholesky factor has sparsity pattern of choice! Here: banded and lower triangular -> efficient!

    preconditioner_inv = cholfac_precision @ cholfac_precision.mT
    if rescale:
        preconditioner_inv = rescale_preconditioner(
            preconditioner_inv=preconditioner_inv, kernel=kernel, noise=noise, X_train=X_train
        )
    return preconditioner_inv
