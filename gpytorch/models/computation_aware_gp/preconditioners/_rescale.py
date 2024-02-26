"""Rescale a preconditioner to be valid for prior augmentation."""

import torch


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
