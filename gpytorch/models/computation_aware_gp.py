#!/usr/bin/env python3
"""Computation-aware Gaussian process."""

import math

import torch
from linear_operator import operators, utils as linop_utils

from .. import kernels, likelihoods, means, settings

from ..distributions import MultivariateNormal
from .exact_gp import ExactGP


class ComputationAwareGP(ExactGP):
    """Computation-aware Gaussian process."""

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        mean_module: "means.Mean",
        covar_module: "kernels.Kernel",
        likelihood: "likelihoods.GaussianLikelihood",
        projection_dim: int,
        initialization: str = "random",
    ):

        # Set number of non-zero action entries such that num_non_zero * projection_dim = num_train_targets
        num_non_zero = train_targets.size(-1) // projection_dim

        super().__init__(
            # Training data is subset to satisfy the requirement: num_non_zero * projection_dim = num_train_targets
            train_inputs[0 : num_non_zero * projection_dim],
            train_targets[0 : num_non_zero * projection_dim],
            likelihood,
        )
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.projection_dim = projection_dim
        self.num_non_zero = num_non_zero
        self.cholfac_gram_SKhatS = None

        non_zero_idcs = torch.arange(
            self.num_non_zero * projection_dim,
            device=train_inputs.device,
        ).reshape(self.projection_dim, -1)

        # Initialization of actions
        if initialization == "random":
            # Random initialization
            self.non_zero_action_entries = torch.nn.Parameter(
                torch.randn_like(
                    non_zero_idcs,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                ).div(math.sqrt(self.num_non_zero))
            )
        elif initialization == "targets":
            # Initialize with training targets
            self.non_zero_action_entries = torch.nn.Parameter(
                train_targets.clone()[: self.num_non_zero * projection_dim].reshape(self.projection_dim, -1)
            )
            self.non_zero_action_entries.div(
                torch.linalg.vector_norm(self.non_zero_action_entries, dim=1).reshape(-1, 1)
            )
        elif initialization == "eigen":
            # Initialize via top eigenvectors of kernel submatrices
            with torch.no_grad():
                X = train_inputs.clone()[0 : num_non_zero * projection_dim].reshape(
                    projection_dim, num_non_zero, train_inputs.shape[-1]
                )
                K_sub_matrices = self.covar_module(X)
                _, evecs = torch.linalg.eigh(K_sub_matrices.to_dense())
            self.non_zero_action_entries = torch.nn.Parameter(evecs[:, -1])
        else:
            raise ValueError(f"Unknown initialization: '{initialization}'.")

        self.actions = (
            operators.BlockSparseLinearOperator(  # TODO: Can we speed this up by allowing ranges as non-zero indices?
                non_zero_idcs=non_zero_idcs,
                blocks=self.non_zero_action_entries,
                size_sparse_dim=self.projection_dim * self.num_non_zero,
            )
        )

    def __call__(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            # In training mode, just return the prior.
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        elif settings.prior_mode.on():
            # Prior mode
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        else:
            # Posterior mode
            if x.ndim == 1:
                x = torch.atleast_2d(x).mT

            # Kernel forward and hyperparameters
            if isinstance(self.covar_module, kernels.ScaleKernel):
                outputscale = self.covar_module.outputscale
                lengthscale = self.covar_module.base_kernel.lengthscale
                kernel_forward_fn = self.covar_module.base_kernel._forward_no_kernel_linop
            else:
                outputscale = 1.0
                lengthscale = self.covar_module.lengthscale
                kernel_forward_fn = self.covar_module._forward_no_kernel_linop

            if self.cholfac_gram_SKhatS is None:
                # If the Cholesky factor of the gram matrix S'(K + noise)S hasn't been precomputed
                # (in the loss function), compute it.
                K_lazy = kernel_forward_fn(
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]),
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, 1, self.num_non_zero, self.train_inputs[0].shape[-1]),
                )
                gram_SKS = (
                    (
                        (K_lazy @ self.actions.blocks.view(self.projection_dim, 1, self.num_non_zero, 1)).squeeze(-1)
                        * self.actions.blocks
                    )
                    .sum(-1)
                    .mul(outputscale)
                )

                StrS_diag = (self.actions.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
                gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)
                self.cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                    gram_SKhatS.to(dtype=torch.float64), upper=False
                )

            # Cross-covariance mapped to the low-dimensional space spanned by the actions: k(x, X)S
            covar_x_train_actions = (
                (
                    kernel_forward_fn(
                        x / lengthscale,
                        (self.train_inputs[0] / lengthscale).view(
                            self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]
                        ),
                    )
                    @ self.actions.blocks.view(self.projection_dim, self.num_non_zero, 1)
                )
                .squeeze(-1)
                .mT.mul(outputscale)
            )

            # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                self.cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # "Projected" training data (with mean correction)
            actions_target = self.actions @ (self.train_targets - self.mean_module(self.train_inputs[0]))

            # Compressed representer weights
            compressed_repr_weights = (
                torch.cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64), self.cholfac_gram_SKhatS, upper=False
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            # (Combined) posterior mean and covariance evaluated at the test point(s)
            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x) - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)

            return MultivariateNormal(mean, covar)
