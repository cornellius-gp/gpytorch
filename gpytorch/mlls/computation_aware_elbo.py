#!/usr/bin/env python3
from __future__ import annotations

import math

import torch
from linear_operator import utils

from .. import kernels

from ..likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from .marginal_log_likelihood import MarginalLogLikelihood


class ComputationAwareELBO(MarginalLogLikelihood):
    """Computation-aware ELBO."""

    def __init__(
        self,
        likelihood: GaussianLikelihood,
        model: "gpytorch.models.ComputationAwareGP",
    ):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise NotImplementedError("Likelihood must be Gaussian for computation-aware inference.")
        super().__init__(likelihood, model)

    def forward(self, outputs_batch: torch.Tensor, targets_batch: torch.Tensor, **kwargs):
        # TODO: don't batch objective
        # TODO: comment properly

        num_actions = self.model.actions_op.shape[-2]

        # Training data
        train_inputs = self.model.train_inputs[0]
        train_targets = self.model.train_targets
        num_train_data = len(train_targets)

        if targets_batch.shape[0] < num_train_data:
            # Batched objective
            is_batched_objective = True
            train_inputs_batch = outputs_batch.lazy_covariance_matrix.x1
            num_train_data_batch = targets_batch.shape[0]
            prior_train_inputs = self.model(train_inputs)
        else:
            # Objective evaluated on entire training data
            is_batched_objective = False
            targets_batch = self.model.train_targets[
                0:num_train_data
            ]  # Account for block structure in sparse actions (N=I*K)
            outputs_batch = outputs_batch[0:num_train_data]
            prior_train_inputs = outputs_batch
            num_train_data_batch = num_train_data

        # Kernel
        if isinstance(self.model.covar_module, kernels.ScaleKernel):
            outputscale = self.model.covar_module.outputscale
            lengthscale = self.model.covar_module.base_kernel.lengthscale
            kernel_forward_fn = self.model.covar_module.base_kernel._forward_no_kernel_linop
        else:
            outputscale = 1.0
            lengthscale = self.model.covar_module.lengthscale
            kernel_forward_fn = self.model.covar_module._forward_no_kernel_linop

        # Gramian S'KS
        del self.model.cholfac_gram_SKhatS  # Explicitly free up memory from prediction

        K_lazy = kernel_forward_fn(
            train_inputs.div(lengthscale).view(
                self.model.projection_dim, self.model.num_non_zero, train_inputs.shape[-1]
            ),
            train_inputs.div(lengthscale).view(
                self.model.projection_dim, 1, self.model.num_non_zero, train_inputs.shape[-1]
            ),
        )
        StK_block_shape = (
            K_lazy @ self.model.actions_op.blocks.view(self.model.projection_dim, 1, self.model.num_non_zero, 1)
        ).squeeze(-1)
        gram_SKS = (StK_block_shape * self.model.actions_op.blocks).sum(-1).mul(outputscale)
        StK = StK_block_shape.view(num_actions, num_actions * self.model.num_non_zero).mul(outputscale)

        StrS_diag = (self.model.actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)

        cholfac_gram_SKhatS = utils.cholesky.psd_safe_cholesky(gram_SKhatS.to(dtype=torch.float64), upper=False)

        # Save Cholesky factor for prediction
        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        if is_batched_objective:
            covar_x_batch_X_train_actions = (
                (
                    kernel_forward_fn(
                        train_inputs_batch.div(lengthscale),
                        train_inputs.div(lengthscale).view(
                            self.model.projection_dim, self.model.num_non_zero, train_inputs.shape[-1]
                        ),
                    )
                    @ self.model.actions_op.blocks.view(self.model.projection_dim, self.model.num_non_zero, 1)
                )
                .squeeze(-1)
                .mT.mul(outputscale)
            )
        else:
            covar_x_batch_X_train_actions = StK.mT

        # Compressed representer weights
        actions_targets = self.model.actions_op._matmul(
            torch.atleast_2d(train_targets - prior_train_inputs.mean).mT
        ).squeeze(-1)
        compressed_repr_weights = torch.cholesky_solve(
            actions_targets.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
        ).squeeze(-1)

        # Expected log-likelihood term
        f_pred_mean_batch = outputs_batch.mean + covar_x_batch_X_train_actions @ torch.atleast_1d(
            compressed_repr_weights
        ).to(dtype=targets_batch.dtype)
        sqrt_downdate = torch.linalg.solve_triangular(
            cholfac_gram_SKhatS, covar_x_batch_X_train_actions.mT, upper=False
        )
        trace_downdate = torch.sum(sqrt_downdate**2, dim=-1)
        f_pred_var_batch = torch.sum(outputs_batch.variance) - torch.sum(trace_downdate)
        expected_log_likelihood_term = -0.5 * (
            num_train_data_batch * torch.log(self.likelihood.noise)
            + 1
            / self.likelihood.noise
            * (torch.linalg.vector_norm(targets_batch - f_pred_mean_batch) ** 2 + f_pred_var_batch)
            + num_train_data_batch * torch.log(torch.as_tensor(2 * math.pi))
        ).div(num_train_data_batch)

        # KL divergence to prior
        kl_prior_term = 0.5 * (
            torch.inner(compressed_repr_weights, (gram_SKS.to(dtype=torch.float64) @ compressed_repr_weights))
            + 2 * torch.sum(torch.log(cholfac_gram_SKhatS.diagonal()))
            - num_actions * torch.log(self.likelihood.noise).to(dtype=torch.float64)
            - torch.log(StrS_diag.to(dtype=torch.float64).sum())
            - torch.trace(torch.cholesky_solve(gram_SKS.to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False))
        ).div(num_train_data)

        elbo = torch.squeeze(expected_log_likelihood_term - kl_prior_term.to(dtype=targets_batch.dtype))
        return elbo
