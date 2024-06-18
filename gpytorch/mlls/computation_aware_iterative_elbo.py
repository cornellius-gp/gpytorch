#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings

import torch
from linear_operator import operators, utils

from .. import kernels

from ..likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from .marginal_log_likelihood import MarginalLogLikelihood


class ComputationAwareIterativeELBO(MarginalLogLikelihood):
    """Computation-aware ELBO."""

    def __init__(
        self,
        likelihood: GaussianLikelihood,
        model: "gpytorch.models.ComputationAwareIterativeGP",
        use_sparse_bilinear_form: bool = False,
    ):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise NotImplementedError("Likelihood must be Gaussian for computation-aware inference.")
        super().__init__(likelihood, model)
        self.use_sparse_bilinear_form = use_sparse_bilinear_form

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        # prior_dist = self.model.preconditioner_augmented_forward(self.model.train_inputs[0]) # TODO: enable prior augmentation

        if self.use_sparse_bilinear_form:
            return self._sparse_forward(output, target, **kwargs)

        K = output.lazy_covariance_matrix
        Khat = self.likelihood(output).lazy_covariance_matrix

        # Linear solve
        solver_state = self.model.linear_solver.solve(Khat, target - output.mean)
        if self.model.prediction_strategy is None:
            self.model._solver_state = solver_state
        else:
            warnings.warn("MLL does not set solver_state during its computation. This could cause undefined behavior.")

        if str(self.model.linear_solver.policy) == "GradientPolicy()":
            # Reorthogonalize actions twice for stability (only applies to IterGP-CG):
            S, _ = torch.linalg.qr(solver_state.cache["actions_op"].mT, mode="reduced")
            S, _ = torch.linalg.qr(S, mode="reduced")
            solver_state.cache["actions_op"] = S.mT

        actions_op = solver_state.cache["actions_op"]
        num_actions = actions_op.shape[0]
        num_train_data = target.shape[0]

        # Gramian S'KS
        K_actions = (
            actions_op._matmul(K).mT
            if isinstance(actions_op, operators.BlockSparseLinearOperator)
            else K @ actions_op.mT
        )
        gram_SKS = (
            actions_op._matmul(actions_op._matmul(K).mT)
            if isinstance(actions_op, operators.BlockSparseLinearOperator)
            else actions_op @ K_actions
        )
        if str(self.model.linear_solver.policy) == "GradientPolicy()":
            # Actions are provably orthogonal (and normalized by assumption)
            StrS = torch.eye(num_actions, device=gram_SKS.device, dtype=gram_SKS.dtype)
        else:
            StrS = (
                actions_op._matmul(actions_op.to_dense().mT)
                if isinstance(actions_op, operators.BlockSparseLinearOperator)
                else actions_op @ actions_op.mT
            )
        gram_SKhatS = gram_SKS + self.likelihood.noise * StrS

        cholfac_gram = utils.cholesky.psd_safe_cholesky(gram_SKhatS, upper=False)

        if str(self.model.linear_solver.policy) == "GradientPolicy()":
            solver_state.cache["cholfac_gram"] = cholfac_gram

        # Compressed representer weights
        actions_target = actions_op @ (target - output.mean)
        compressed_repr_weights = torch.cholesky_solve(actions_target.unsqueeze(1), cholfac_gram, upper=False).squeeze(
            -1
        )
        if str(self.model.linear_solver.policy) == "GradientPolicy()":
            solver_state.cache["compressed_solution"] = compressed_repr_weights

        # Expected log-likelihood term
        f_pred_mean = output.mean + K_actions @ torch.atleast_1d(compressed_repr_weights)
        sqrt_downdate = torch.linalg.solve_triangular(cholfac_gram, K_actions.mT, upper=False)
        trace_downdate = torch.sum(sqrt_downdate**2, dim=0)
        f_pred_var = torch.sum(output.variance) - torch.sum(trace_downdate)
        expected_log_likelihood_term = -0.5 * (
            num_train_data * torch.log(self.likelihood.noise)
            + 1 / self.likelihood.noise * (torch.linalg.vector_norm(target - f_pred_mean) ** 2 + f_pred_var)
            + num_train_data * torch.log(torch.as_tensor(2 * math.pi))
        )

        # KL divergence to prior
        kl_prior_term = 0.5 * (
            torch.inner(compressed_repr_weights, (gram_SKS @ compressed_repr_weights))
            + 2 * torch.sum(torch.log(cholfac_gram.diagonal()))
            - num_actions * torch.log(self.likelihood.noise)
            - torch.logdet(StrS)
            - torch.trace(torch.cholesky_solve(gram_SKS, cholfac_gram, upper=False))
        )

        elbo = torch.squeeze(expected_log_likelihood_term - kl_prior_term)
        return elbo.div(num_train_data)

    def _sparse_forward(self, outputs_batch: torch.Tensor, targets_batch: torch.Tensor, **kwargs):

        actions_op = self.model.actions
        num_actions = actions_op.shape[0]

        # Training data batch
        train_inputs_batch = outputs_batch.lazy_covariance_matrix.x1
        num_train_data_batch = targets_batch.shape[0]
        num_train_data = len(self.model.train_inputs[0])

        # Kernel
        kernel = self.model.covar_module
        if isinstance(kernel, kernels.ScaleKernel):
            outputscale = kernel.outputscale
            lengthscale = kernel.base_kernel.lengthscale
            kernel_forward_fn = kernel.base_kernel._forward_no_kernel_linop
            # kernel_vjp_fn = lambda V, X1, X2: kernel.base_kernel._forward_and_vjp(X1, X2, V)[1]
            # kernel_forward_and_vjp_fn = kernel.base_kernel._forward_and_vjp
            base_kernel = kernel.base_kernel
        else:
            outputscale = 1.0
            lengthscale = kernel.lengthscale
            kernel_forward_fn = kernel._forward_no_kernel_linop
            # kernel_forward_and_vjp_fn = kernel.base_kernel._forward_and_vjp
            # kernel_vjp_fn = kernel._vjp
            base_kernel = kernel

        # Prior mean and kernel
        train_targets = self.model.train_targets
        # if num_train_data > num_train_data_batch:
        prior_mean = self.model.mean_module(self.model.train_inputs[0])
        # else:
        #     prior_mean = outputs_batch.mean
        # TODO: We can optimize this if we don't batch to avoid one evaluation of the prior mean

        # Gramian S'KS
        del self.model.cholfac_gram_SKhatS  # Explicitly free up memory from prediction
        # gram_SKS = kernels.SparseQuadForm.apply(
        #     self.model.train_inputs[0] / lengthscale,
        #     actions_op.blocks,
        #     None,
        #     # actions_op.non_zero_idcs.mT,
        #     kernel_forward_fn,
        #     kernel_forward_and_vjp_fn,
        #     # self.model.chunk_size,
        # )
        K_lazy = kernel_forward_fn(
            self.model.train_inputs[0]
            .div(lengthscale)
            .view(self.model.num_iter, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
            self.model.train_inputs[0]
            .div(lengthscale)
            .view(self.model.num_iter, 1, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
        )
        gram_SKS = (
            (
                (K_lazy @ actions_op.blocks.view(self.model.num_iter, 1, self.model.num_non_zero, 1)).squeeze(-1)
                * actions_op.blocks
            )
            .sum(-1)
            .mul(outputscale)
        )

        StrS_diag = (actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)

        cholfac_gram_SKhatS = utils.cholesky.psd_safe_cholesky(gram_SKhatS.to(dtype=torch.float64), upper=False)
        # Save Cholesky factor for prediction
        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        # covar_x_batch_X_train_actions = outputscale * kernels.SparseLinearForm.apply(
        #     train_inputs_batch / lengthscale,
        #     self.model.train_inputs[0] / lengthscale,
        #     actions_op.blocks,
        #     actions_op.non_zero_idcs,
        #     kernel_forward_fn,
        #     kernel_forward_and_vjp_fn,
        #     # self.model.chunk_size,
        # )  # TODO: Set chunk size differently here, since we are only allocating 0.1*n^2 here typically

        covar_x_batch_X_train_actions = (
            (
                kernel_forward_fn(
                    train_inputs_batch.div(lengthscale),
                    self.model.train_inputs[0]
                    .div(lengthscale)
                    .view(self.model.num_iter, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
                )
                @ actions_op.blocks.view(self.model.num_iter, self.model.num_non_zero, 1)
            )
            .squeeze(-1)
            .mT.mul(outputscale)
        )

        # Compressed representer weights
        actions_targets = actions_op._matmul(torch.atleast_2d(train_targets - prior_mean).mT).squeeze(-1)
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
