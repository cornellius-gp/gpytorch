"""Preconditioners for kernel matrices."""

import abc

import numpy as np
import torch
from linear_operator.operators import DiagLinearOperator
from torch import nn

from .... import settings


# class Preconditioner(nn.Module, abc.ABC):
#     def __init__(self, kernel, noise, X, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.kernel = kernel
#         self.noise = noise
#         self.X = X


# class Scalar(Preconditioner):

#     def __init__(self, kernel, noise, X, **kwargs) -> None:
#         super().__init__(kernel=kernel, noise=noise, X=X, **kwargs)
#         self.lmda_max_upper_bound = torch.max(torch.sum(self.kernel(self.X), dim=1))

#     def forward(self, x):
#         return DiagLinearOperator(1 / (self.lmda_max_upper_bound + self.noise) * torch.ones(self.X.shape[0])) @ x
# TODO: need a caching mechanism so we only recompute the preconditioner when the hyperparameters are updated


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


# class SparsePrecision(Preconditioner):
#     def __init__(self, kernel, noise, X, num_diags_cholfac: int, **kwargs) -> None:
#         super().__init__(kernel=kernel, noise=noise, X=X, **kwargs)
#         self.num_diags_cholfac = num_diags_cholfac

#     def forward(self, x):
#         DiagLinearOperator(1 / (self.lmda_max_upper_bound + self.noise) * torch.ones(self.X.shape[0])) @ x


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

        unit_vec = torch.zeros((len(sparsity_set), 1))
        unit_vec[0, 0] = 1.0

        cholfac_K_neighbor_set_unit_vector = torch.cholesky_solve(unit_vec, cholfac_K_sparsity_set, upper=False)
        cholfac_precision[i : np.minimum(i + num_diags_cholfac, X_train.shape[0]), i] = (
            cholfac_K_neighbor_set_unit_vector / (unit_vec.mT @ cholfac_K_neighbor_set_unit_vector).item()
        ).reshape(
            -1
        )  # Note: Cholesky factor has sparsity pattern of choice! Here: banded and lower triangular -> efficient!

    preconditioner_inv = cholfac_precision @ cholfac_precision.mT
    if rescale:
        preconditioner_inv = rescale_preconditioner(
            preconditioner_inv=preconditioner_inv, kernel=kernel, noise=noise, X_train=X_train
        )
    return preconditioner_inv


def polynomial():
    raise NotImplementedError


def low_rank_plus_diagonal(low_rank_factor, diag_const):
    # Matrix inversion lemma
    raise NotImplementedError


def rescale_preconditioner(
    preconditioner_inv,
    kernel,
    noise,
    X_train,
):
    # Ensure preconditioner validity
    upper_bound_max_eval_preconditioner_inv = torch.max(torch.sum(torch.abs(preconditioner_inv), dim=1))
    upper_bound_max_eval_Khat = torch.sum(kernel(X_train)) / X_train.shape[0] + noise
    lower_bound_max_eval_Khat = torch.max(
        torch.sum(
            torch.abs(kernel(X_train).to_dense() + noise * torch.eye(X_train.shape[0])),
            dim=1,
        )
    )
    scalar_factor_precond_inv = (
        torch.minimum(lower_bound_max_eval_Khat, 1.0 / upper_bound_max_eval_Khat)
        / upper_bound_max_eval_preconditioner_inv
    )
    print(scalar_factor_precond_inv)
    return scalar_factor_precond_inv * preconditioner_inv
