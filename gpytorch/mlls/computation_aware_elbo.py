#!/usr/bin/env python3
from __future__ import annotations

import math

import torch
from linear_operator import utils

from .. import kernels, settings

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

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs):

        # Initialize some useful variables
        train_inputs = self.model.train_inputs[0]
        train_targets = self.model.train_targets
        num_train_data = len(train_targets)
        prior_evaluated_at_train_inputs = outputs[
            0:num_train_data
        ]  # Training data size might not exactly equal NNZ * PROJ_DIM

        if settings.debug.on():
            # Check whether training objective is evaluated at the training data
            # Note that subsetting is needed here, since a block sparse projection with equal block size
            # necessitates that num_train_data = NNZ * PROJ_DIM
            if (not torch.equal(train_inputs, outputs.lazy_covariance_matrix.x1[0:num_train_data])) or (
                not torch.equal(train_targets, targets[0:num_train_data])
            ):
                raise RuntimeError("You must evaluate the objective on the training inputs!")

        # Kernel
        if isinstance(self.model.covar_module, kernels.ScaleKernel):
            outputscale = self.model.covar_module.outputscale
            lengthscale = self.model.covar_module.base_kernel.lengthscale
            kernel_forward_fn = self.model.covar_module.base_kernel._forward_no_kernel_linop
        else:
            outputscale = 1.0
            lengthscale = self.model.covar_module.lengthscale
            kernel_forward_fn = self.model.covar_module._forward_no_kernel_linop

        # Explicitly free up memory from prediction to avoid unnecessary memory overhead
        # TODO: does this really do much if we zero the gradients anyway?
        del self.model.cholfac_gram_SKhatS

        # Lazily evaluate kernel at training inputs as a 4D tensor with shape (PROJ_DIM, PROJ_DIM, NNZ, NNZ)
        K_lazy = kernel_forward_fn(
            train_inputs.div(lengthscale).view(
                self.model.projection_dim, self.model.num_non_zero, train_inputs.shape[-1]
            ),
            train_inputs.div(lengthscale).view(
                self.model.projection_dim, 1, self.model.num_non_zero, train_inputs.shape[-1]
            ),
        )

        # Compute S'K in block shape (PROJ_DIM, PROJ_DIM, NNZ)
        StK_block_shape = (
            K_lazy @ self.model.actions_op.blocks.view(self.model.projection_dim, 1, self.model.num_non_zero, 1)
        ).squeeze(-1)
        covar_x_batch_X_train_actions = (
            StK_block_shape.view(self.model.projection_dim, self.model.projection_dim * self.model.num_non_zero)
            .mul(outputscale)
            .mT
        )

        # Projected Gramians S'KS and S'(K + noise)S
        gram_SKS = (StK_block_shape * self.model.actions_op.blocks).sum(-1).mul(outputscale)
        StrS_diag = (self.model.actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)

        # Cholesky factor of Gramian
        cholfac_gram_SKhatS = utils.cholesky.psd_safe_cholesky(gram_SKhatS.to(dtype=torch.float64), upper=False)

        # Save Cholesky factor for prediction
        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        # "Projected" training data (with mean correction)
        actions_targets = self.model.actions_op._matmul(
            torch.atleast_2d(train_targets - prior_evaluated_at_train_inputs.mean).mT
        ).squeeze(-1)

        # Compressed representer weights
        compressed_repr_weights = torch.cholesky_solve(
            actions_targets.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
        ).squeeze(-1)

        # Expected log-likelihood term
        f_pred_mean_batch = prior_evaluated_at_train_inputs.mean + covar_x_batch_X_train_actions @ torch.atleast_1d(
            compressed_repr_weights
        ).to(dtype=targets.dtype)
        sqrt_downdate = torch.linalg.solve_triangular(
            cholfac_gram_SKhatS, covar_x_batch_X_train_actions.mT, upper=False
        )
        trace_downdate = torch.sum(sqrt_downdate**2, dim=-1)
        f_pred_var_batch = torch.sum(prior_evaluated_at_train_inputs.variance) - torch.sum(trace_downdate)
        expected_log_likelihood_term = -0.5 * (
            num_train_data * torch.log(self.likelihood.noise)
            + 1
            / self.likelihood.noise
            * (torch.linalg.vector_norm(train_targets - f_pred_mean_batch) ** 2 + f_pred_var_batch)
            + num_train_data * torch.log(torch.as_tensor(2 * math.pi))
        ).div(num_train_data)

        # KL divergence to prior
        kl_prior_term = 0.5 * (
            torch.inner(compressed_repr_weights, (gram_SKS.to(dtype=torch.float64) @ compressed_repr_weights))
            + 2 * torch.sum(torch.log(cholfac_gram_SKhatS.diagonal()))
            - self.model.projection_dim * torch.log(self.likelihood.noise).to(dtype=torch.float64)
            - torch.log(StrS_diag.to(dtype=torch.float64).sum())
            - torch.trace(torch.cholesky_solve(gram_SKS.to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False))
        ).div(num_train_data)

        elbo = torch.squeeze(expected_log_likelihood_term - kl_prior_term.to(dtype=targets.dtype))
        return elbo
