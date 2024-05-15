#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Callable, Optional, Tuple, Union

import torch
from jaxtyping import Float
from linear_operator import operators, utils
from pykeops.torch import LazyTensor

from .. import kernels

from ..likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from .marginal_log_likelihood import MarginalLogLikelihood


class ComputationAwareMarginalLogLikelihoodAutoDiff(MarginalLogLikelihood):
    """Computation-aware marginal log-likelihood with gradients via automatic differentiation."""

    def __init__(
        self, likelihood: GaussianLikelihood, model: "ComputationAwareGP", use_sparse_bilinear_form: bool = False
    ):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference.")
        super().__init__(likelihood, model)
        self.use_sparse_bilinear_form = use_sparse_bilinear_form

    def _forward_sparse_bilinear_form(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        # Kernel linear operator
        Khat = self.likelihood(output).lazy_covariance_matrix

        # Linear solve
        solver_state = self.model.linear_solver.solve(
            Khat,
            target - output.mean,
            train_inputs=self.model.train_inputs[0],
            kernel=self.model.covar_module,
            noise=self.likelihood.noise,
        )  # TODO: do we really need to do a solve here? Seems like wasted compute since we only need this to get the actions.
        if self.model.prediction_strategy is None:
            self.model._solver_state = solver_state
        else:
            warnings.warn("MLL does not set solver_state during its computation. This could cause undefined behavior.")

        actions_op = solver_state.cache["actions_op"]
        actions_op.requires_grad = False  # TODO: taking gradients here is really slow, why?
        num_actions = actions_op.shape[0]

        # Kernel parameters
        outputscale = 1.0
        lengthscale = 1.0
        noise = self.likelihood.noise
        if isinstance(self.model.covar_module, kernels.ScaleKernel):
            outputscale = self.model.covar_module.outputscale
            lengthscale = self.model.covar_module.base_kernel.lengthscale
            forward_fn = self.model.covar_module.base_kernel._forward
            vjp_fn = self.model.covar_module.base_kernel._vjp
        else:
            try:
                lengthscale = self.model.covar_module.lengthscale
            except AttributeError:
                pass

            forward_fn = self.model.covar_module._forward
            vjp_fn = self.model.covar_module._vjp

        SKS, SS = kernels.SparseBilinearForms.apply(
            self.model.train_inputs[0] / lengthscale,
            actions_op.blocks.mT,
            actions_op.blocks.mT,
            actions_op.non_zero_idcs.mT,
            actions_op.non_zero_idcs.mT,
            forward_fn,
            vjp_fn,
            None if self.model.train_inputs[0].shape[0] < 10000 else 1,
        )
        L = torch.linalg.cholesky(
            outputscale * SKS + noise * SS,  # + 1e-5 * torch.eye(num_actions, dtype=SS.dtype, device=SS.device),
            upper=False,
        )  # TODO: do we really need this extra nugget?
        Sy = actions_op @ target
        compressed_rweights = torch.cholesky_solve(Sy.unsqueeze(1), L, upper=False).squeeze()
        lml = -(
            0.5 * torch.inner(Sy, compressed_rweights)
            + torch.sum(torch.log(L.diagonal()))
            + 0.5 * num_actions * math.log(2 * math.pi)
        ).div(num_actions)
        return lml

    def _forward_sparse_linop(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        # Kernel linear operator
        Khat = self.likelihood(output).lazy_covariance_matrix.evaluate_kernel()

        # Linear solve
        solver_state = self.model.linear_solver.solve(Khat, target - output.mean)
        if self.model.prediction_strategy is None:
            self.model._solver_state = solver_state
        else:
            warnings.warn("MLL does not set solver_state during its computation. This could cause undefined behavior.")

        # Compute the log marginal likelihood
        actions_op = solver_state.cache["actions_op"]
        num_actions = actions_op.shape[0]

        # Gramian S'KS
        if self.model.linear_solver.use_sparse_bilinear_form:
            Khat_subset = Khat[
                solver_state.cache["actions_op"].non_zero_idcs[:, :, None, None],
                solver_state.cache["actions_op"].non_zero_idcs[None, None, :, :],
            ]
            gram_SKS = (
                solver_state.cache["actions_op"].blocks
                * (Khat_subset * solver_state.cache["actions_op"].blocks).sum((-2, -1))
            ).sum(-1)
        else:
            # TODO: Can we accelerate this by just caching SKS during the solver run *with gradients* and then using it?
            # Or even cholfac_gram directly? => would also avoid Cholesky throwing an error about the leading minor being
            # not positive-definite
            # Should avoid an extra O(nik) operation
            gram_SKS = (
                actions_op._matmul(actions_op._matmul(Khat).mT)
                if isinstance(actions_op, operators.BlockSparseLinearOperator)
                else actions_op @ (actions_op @ Khat).mT
            )

        cholfac_gram = torch.linalg.cholesky(
            gram_SKS + torch.eye(num_actions, dtype=gram_SKS.dtype, device=gram_SKS.device) * 1e-5, upper=False
        )  # TODO: do we really need the nugget here?

        # Compressed representer weights
        actions_target = actions_op @ (target - output.mean)
        compressed_repr_weights = torch.cholesky_solve(actions_target.unsqueeze(1), cholfac_gram, upper=False).squeeze()

        # TODO: Avoid the O(ik) operation here
        # Instead: Cache S'y in the linear solver state and compute y'S @ compr_repr_weights
        if self.model.preconditioner is None:
            lml = -(
                0.5 * torch.inner(actions_target, compressed_repr_weights)
                + torch.sum(torch.log(cholfac_gram.diagonal()))
                + 0.5 * num_actions * math.log(2 * math.pi)
            )
            # Normalize log-marginal likelihood
            normalized_lml = lml.div(num_actions)
        else:
            # # LML via preconditioner only
            # # TODO: REMOVE ME?
            # warnings.warn("Computing LML via preconditioner only!")

            # num_data = target.shape[0]
            # cholfac_Pinv = torch.linalg.cholesky(
            #     self.model.preconditioner.inv_matmul(
            #         torch.eye(num_data),
            #         kernel=self.model.covar_module,
            #         noise=self.likelihood.noise,
            #         X=self.model.train_inputs[0],
            #     )
            # )
            # target_minus_prior_mean = target - self.model.mean_module(self.model.train_inputs[0])
            # lml = -(
            #     0.5
            #     * torch.inner(
            #         target_minus_prior_mean,
            #         self.model.preconditioner.inv_matmul(
            #             target_minus_prior_mean,
            #             kernel=self.model.covar_module,
            #             noise=self.likelihood.noise,
            #             X=self.model.train_inputs[0],
            #         ),
            #     )
            #     + torch.sum(torch.log(cholfac_Pinv.diagonal()))
            #     + 0.5 * num_data * math.log(2 * math.pi)
            # )
            # # Normalize log-marginal likelihood
            # normalized_lml = lml.div(num_data)

            # LML for preconditioner-augmented prior
            warnings.warn("Computing LML via preconditioner-augmented prior only!")
            prior_dist = self.model.preconditioner_augmented_forward(self.model.train_inputs[0])
            prior_mean = prior_dist.mean
            prior_pred_cov = self.likelihood(prior_dist).lazy_covariance_matrix.evaluate_kernel().to_dense()
            num_data = target.shape[0]

            cholfac_prior_pred_cov = torch.linalg.cholesky(prior_pred_cov, upper=False)
            lml = -(
                0.5
                * torch.inner(
                    target - prior_mean,
                    torch.cholesky_solve(
                        (target - prior_mean).unsqueeze(-1), cholfac_prior_pred_cov, upper=False
                    ).squeeze(),
                )
                + torch.sum(torch.log(cholfac_prior_pred_cov.diagonal()))
                + 0.5 * num_data * math.log(2 * math.pi)
            )
            # Normalize log-marginal likelihood
            normalized_lml = lml.div(num_data)

        return normalized_lml

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        if self.use_sparse_bilinear_form:
            return self._forward_sparse_bilinear_form(output=output, target=target, **kwargs)
        else:
            return self._forward_sparse_linop(output=output, target=target, **kwargs)


class ComputationAwareMarginalLogLikelihood(MarginalLogLikelihood):
    """
    Computation-aware marginal log-likelihood for a Gaussian process with a computation-aware Gaussian likelihood.


    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:
        >>> mll = gpytorch.mlls.ComputationAwareMarginalLogLikelihood(likelihood, model)
        # TODO
    """

    def __init__(self, likelihood: GaussianLikelihood, model: "ComputationAwareGP"):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference.")
        super().__init__(likelihood, model)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        Khat = self.likelihood(output).lazy_covariance_matrix.evaluate_kernel()

        # with torch.no_grad():
        solver_state = self.model.linear_solver.solve(Khat, target)  # TODO: do we need to subtract output.mean here?
        if self.model.prediction_strategy is None:
            self.model._solver_state = solver_state

        compressed_repr_weights = solver_state.cache["compressed_solution"]
        actions_op = solver_state.cache["actions_op"]
        cholfac_gram = solver_state.cache["cholfac_gram"]
        logdet_estimate = solver_state.logdet
        residual = solver_state.residual
        policy_hyperparams = list(self.model.linear_solver.policy.parameters())

        if len(policy_hyperparams) > 1:
            raise ValueError("Cannot define a policy with more than one tensor hyperparameter.")
        elif len(policy_hyperparams) == 1:
            policy_hyperparams = policy_hyperparams[0]
        else:
            policy_hyperparams = None

        # Implementing this via an autograd function is the recommended pattern by
        # PyTorch for extending nn.Module with a custom backward pass.
        # See also: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-nn
        return _ComputationAwareMarginalLogLikelihoodFunction.apply(
            Khat.representation_tree(),
            target,
            compressed_repr_weights,
            actions_op,
            cholfac_gram,
            logdet_estimate,
            residual,
            policy_hyperparams,
            *Khat.representation(),
        )


class _ComputationAwareMarginalLogLikelihoodFunction(torch.autograd.Function):
    """Autograd function computing the computation-aware marginal log-likelihood."""

    @staticmethod
    def forward(
        ctx,
        Khat_representation_tree,
        target: torch.Tensor,
        compressed_repr_weights: torch.Tensor,
        actions_op: Union[operators.LinearOperator, torch.Tensor],
        cholfac_gram: torch.Tensor,
        logdet_estimate: torch.Tensor,
        residual: torch.Tensor,
        policy_hyperparams: torch.Tensor,
        *linear_op_representation: Tuple[torch.Tensor],
    ):
        # Reconstruct Khat from representation tree
        Khat = Khat_representation_tree(*linear_op_representation)

        # Log marginal likelihood
        num_actions = compressed_repr_weights.shape[0]

        # TODO: Avoid the O(ik) operation here
        # Instead: Cache S'y in the linear solver state and compute y'S @ compr_repr_weights
        lml = -0.5 * (
            torch.inner(actions_op @ target, compressed_repr_weights)
            + logdet_estimate
            + num_actions * math.log(2 * math.pi)
        )

        ctx.Khat = Khat
        ctx.compressed_repr_weights = compressed_repr_weights
        ctx.actions_op = actions_op
        ctx.cholfac_gram = cholfac_gram
        ctx.residual = residual
        ctx.policy_hyperparams = policy_hyperparams

        normalized_lml = lml.div(num_actions)

        return normalized_lml

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            _custom_gradient_wrt_policy_hyperparameters(
                Khat=ctx.Khat,
                policy_hyperparams=ctx.policy_hyperparams,
                actions_op=ctx.actions_op,
                compressed_repr_weights=ctx.compressed_repr_weights,
                residual=ctx.residual,
                cholfac_gram=ctx.cholfac_gram,
            ),
            *_custom_gradient_wrt_kernel_hyperparameters(
                Khat=ctx.Khat,
                actions_op=ctx.actions_op,
                compressed_repr_weights=ctx.compressed_repr_weights,
                cholfac_gram=ctx.cholfac_gram,
            ),
        )


def _custom_gradient_wrt_policy_hyperparameters(
    Khat: operators.LinearOperator,
    policy_hyperparams: torch.Tensor,
    actions_op: Union[operators.LinearOperator, torch.Tensor],
    compressed_repr_weights: torch.Tensor,
    residual: torch.Tensor,
    cholfac_gram: torch.Tensor,
) -> Union[None, torch.Tensor]:
    # No gradients required
    if policy_hyperparams is None:
        return None
    if not policy_hyperparams.requires_grad:
        return None

    # Gradient of negative LML with respect to policy hyperparameters
    with torch.no_grad():
        actions_linear_op = (
            actions_op._matmul(Khat)
            if isinstance(actions_op, operators.BlockSparseLinearOperator)
            else actions_op @ Khat
        )

    with torch.set_grad_enabled(True):
        # Explicit gradient with S instead of dS/dparams that is linear in S
        gradient_helper = -torch.sum(  # TODO: should there be a minus here?
            (
                torch.outer(residual, compressed_repr_weights).mT
                + torch.cholesky_solve(actions_linear_op, cholfac_gram, upper=False)
            )
            * (
                actions_op.to_dense()
                if isinstance(actions_op, operators.BlockSparseLinearOperator)  # TODO:  Allocates unnecessary memory
                else actions_op
            )
        )

    # Take gradient of the helper function to get overall gradient of LML with respect to policy hyperparameters
    gradient_helper.backward(retain_graph=True)  # TODO: is this hacky?

    return policy_hyperparams.grad


def _custom_gradient_wrt_kernel_hyperparameters(
    Khat: operators.LinearOperator,
    actions_op: Union[operators.LinearOperator, torch.Tensor],
    compressed_repr_weights: torch.Tensor,
    cholfac_gram: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    args = tuple(Khat.representation())
    args_with_grads = tuple(arg for arg in args if arg.requires_grad)

    # No gradients required
    if not len(args_with_grads):
        return tuple(None for _ in args)

    # Gradient of LML with respect to kernel hyperparameters
    def _neg_normalized_lml_gradient_helper(*representation):
        """Helper function to compute gradient of LML with respect to kernel hyperparameters.

        Implements the negative normalized gradient as a linear function of K with K instead of dK/dtheta. Then
        the gradient can be taken by taking the gradient of this function with respect to theta.
        """
        lin_op_copy = Khat.representation_tree()(*representation)
        gram_SKS = (
            actions_op._matmul(actions_op._matmul(lin_op_copy).mT)
            # TODO: we can possibly do better here by pulling the diagonal matrix out: S'diag S
            # => avoids extra O(ni) memory
            if isinstance(actions_op, operators.BlockSparseLinearOperator)
            else actions_op @ (actions_op @ lin_op_copy).mT
        )
        # Data fit term "gradient" with K instead of dK/dtheta
        quadratic_loss_term = torch.inner(compressed_repr_weights, gram_SKS @ compressed_repr_weights)

        # Complexity term "gradient" with K instead of dK/dtheta
        complexity_term = torch.trace(torch.cholesky_solve(gram_SKS, cholfac_gram, upper=False))

        return -0.5 * (quadratic_loss_term - complexity_term).div(compressed_repr_weights.shape[0])

    # Compute gradient of LML with respect to kernel hyperparameters
    actual_grads = deque(torch.autograd.functional.vjp(_neg_normalized_lml_gradient_helper, Khat.representation())[1])

    # Now make sure that the object we return has one entry for every item in args
    grads = []
    for arg in args:
        if arg.requires_grad:
            grads.append(actual_grads.popleft())
        else:
            grads.append(None)

    return tuple(grads)


class ComputationAwareELBO(MarginalLogLikelihood):
    """Computation-aware ELBO."""

    def __init__(
        self, likelihood: GaussianLikelihood, model: "ComputationAwareGP", use_sparse_bilinear_form: bool = False
    ):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference.")
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

        # # Idea: Subsample set of actions used
        # action_idcs = torch.randperm(self.model.actions.shape[0])[0:32]
        # actions_op = operators.BlockSparseLinearOperator(
        #     non_zero_idcs=self.model.actions.non_zero_idcs[action_idcs, :],
        #     blocks=self.model.actions.blocks[action_idcs, :],
        #     size_sparse_dim=self.model.actions.size_sparse_dim,
        # )
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


class ComputationAwareELBOCustomBackward(MarginalLogLikelihood):
    """Computation-aware ELBO."""

    def __init__(
        self, likelihood: GaussianLikelihood, model: "ComputationAwareGP", num_blocks_with_grad: Optional[int] = None
    ):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference.")
        super().__init__(likelihood, model)
        self.num_blocks_with_grad = num_blocks_with_grad

    def forward(self, outputs_batch: torch.Tensor, targets_batch: torch.Tensor, **kwargs):

        # Actions
        actions_op = self.model.actions
        num_actions = actions_op.shape[-2]

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
            # Objective evaluated on entire trainign data
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

        if self.num_blocks_with_grad is None:
            K_lazy = kernel_forward_fn(
                self.model.train_inputs[0]
                .div(lengthscale)
                .view(self.model.num_iter, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
                self.model.train_inputs[0]
                .div(lengthscale)
                .view(self.model.num_iter, 1, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
            )
            StK_block_shape = (
                K_lazy @ actions_op.blocks.view(self.model.num_iter, 1, self.model.num_non_zero, 1)
            ).squeeze(-1)
            gram_SKS = (StK_block_shape * actions_op.blocks).sum(-1).mul(outputscale)
            StK = StK_block_shape.view(num_actions, num_actions * self.model.num_non_zero).mul(outputscale)
        else:
            gram_SKS, StK = self._compute_StKS_stochastic_gradient(
                X=train_inputs.div(lengthscale),
                Sv=actions_op.blocks,
                kernel=kernel_forward_fn,
                num_blocks_with_grad=self.num_blocks_with_grad,
            )
            gram_SKS = gram_SKS.mul(outputscale)
            StK = StK.mul(outputscale)

        StrS_diag = (actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)

        cholfac_gram_SKhatS = utils.cholesky.psd_safe_cholesky(gram_SKhatS.to(dtype=torch.float64), upper=False)

        # Save Cholesky factor for prediction
        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        if is_batched_objective:
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
        else:
            covar_x_batch_X_train_actions = StK.mT

        # Compressed representer weights
        actions_targets = actions_op._matmul(torch.atleast_2d(train_targets - prior_train_inputs.mean).mT).squeeze(-1)
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

    def _compute_StKS_stochastic_gradient(
        self,
        X: Float[torch.Tensor, "... N D"],
        Sv: Float[torch.Tensor, "... I K"],
        kernel: Callable[
            [Float[torch.Tensor, "... M1 D"], Float[torch.Tensor, "... M2 D"]],
            Float[LazyTensor, "... M1 M2"],
        ],
        num_blocks_with_grad: Optional[int] = None,
    ) -> Float[torch.Tensor, "... I I"]:
        """
        num_blocks_with_grad: if we divide S^T K S into rows, how many of the rows do we want contributing
            to the gradient
        """

        *X_batch_shape, N, D = X.shape
        *Sv_batch_shape, I, K = Sv.shape

        if num_blocks_with_grad is None:
            num_blocks_with_grad = I
        num_blocks_no_grad = I - num_blocks_with_grad

        Sv_detached = Sv

        # Randomly shuffle X blocks and Sv blocks
        # This ensures that we are randomly detaching parts of X for the stochastic gradient computation
        sv_order = torch.randperm(I, device=Sv.device)
        arange = torch.arange(K, device=Sv.device)
        x_order = (sv_order.mul(K)[:, None] + arange).view(N)
        Sv_detached_perm = Sv_detached  # [..., sv_order, :]
        X_perm = X[..., x_order, :]

        # Detach parts of X
        X_with_grad = X_perm[..., : num_blocks_with_grad * K, :]
        if X.requires_grad:
            X_with_grad.register_hook(lambda grad: grad * (I / float(num_blocks_with_grad)))
            # Up-weight the stochastic gradient terms based on how much we are dropping out
        X_no_grad = X_perm[..., num_blocks_with_grad * K :, :].detach()

        # Compute StK
        K_nonzero_with_grad = kernel(
            X_with_grad.view(*X_batch_shape, 1, num_blocks_with_grad, K, D),
            X.view(*X_batch_shape, I, 1, K, D),
        )
        StK_nonzero_with_grad = (K_nonzero_with_grad @ Sv_detached_perm.view(*Sv_batch_shape, I, 1, K, 1)).squeeze(-1)
        K_nonzero_no_grad = kernel(
            X_no_grad.view(*X_batch_shape, 1, num_blocks_no_grad, K, D),
            X.view(*X_batch_shape, I, 1, K, D).detach(),
        )
        StK_nonzero_no_grad = (K_nonzero_no_grad @ Sv_detached_perm.view(*Sv_batch_shape, I, 1, K, 1)).squeeze(-1)
        StK_nonzero = torch.cat([StK_nonzero_with_grad, StK_nonzero_no_grad], dim=-2)

        # Unshuffle the entries of StK
        orig_sv_order = torch.empty_like(sv_order)
        orig_sv_order[sv_order] = torch.arange(sv_order.size(0), device=sv_order.device)
        StK_nonzero = StK_nonzero[..., :, orig_sv_order, :]

        # Now compute StKS
        StKS = torch.einsum("...ijk,...jk->...ij", StK_nonzero, Sv_detached)

        return StKS, StK_nonzero.view(I, I * K)

    # class GetSvGradThroughStKS(torch.autograd.Function):
    #     """
    #     Autograd hack function so that we can populate Sv's gradient with an analytic computation.
    #     All of the computation on StKS and StKS_nonzero should be performed on a detached version of Sv,
    #     so that Sv won't get gradients except through this function.

    #     The forward pass is a no-op.
    #     The backward pass is also a no-op, except that Sv's gradient gets populated with the analytic gradient computation.
    #     """

    #     @staticmethod
    #     def forward(
    #         ctx,
    #         StKS: Float[torch.Tensor, "... I I"],
    #         StK_nonzero: Float[torch.Tensor, "... I I K"],
    #         Sv: Float[torch.Tensor, "... I K"],
    #     ) -> Float[torch.Tensor, "... I I"]:
    #         ctx.save_for_backward(StK_nonzero)
    #         return StKS

    #     @staticmethod
    #     def backward(
    #         ctx,
    #         grad_output: Float[torch.Tensor, "... I I"],
    #     ) -> Float[torch.Tensor, "... I I"]:
    #         (StK_nonzero,) = ctx.saved_tensors
    #         *Sv_batch_shape, I, _, K = StK_nonzero.shape
    #         N = I * K

    #         StK: Float[torch.Tensor, "... I N"] = StK_nonzero.view(*Sv_batch_shape, I, N)
    #         S_grad = grad_output @ StK + grad_output.mT @ StK
    #         Sv_grad = S_grad[
    #             ...,
    #             torch.arange(I, device=StK.device).view(I, 1),
    #             torch.arange(N, device=StK.device).view(I, K),
    #         ]
    #         return grad_output, None, Sv_grad

    # def _compute_StKS(
    #     self,
    #     X: Float[torch.Tensor, "... N D"],
    #     Sv: Float[torch.Tensor, "... I K"],
    #     kernel: Callable[
    #         [Float[torch.Tensor, "... M1 D"], Float[torch.Tensor, "... M2 D"]],
    #         Float[LazyTensor, "... M1 M2"],
    #     ],
    # ) -> Float[torch.Tensor, "... I I"]:
    #     *X_batch_shape, N, D = X.shape
    #     *Sv_batch_shape, I, K = Sv.shape
    #     Sv_detached = Sv.detach()
    #     # ^^^ note we are purposefully detaching Sv here because we can compute its gradient easily and analytically
    #     # We will use an autograd hack to populate Sv.grad down below
    #     K_nonzero: Float[LazyTensor, "... I I K K"] = kernel(
    #         X.view(*X_batch_shape, 1, I, K, D), X.view(*X_batch_shape, I, 1, K, D)
    #     )
    #     StK_nonzero: Float[torch.Tensor, "... I I K"] = (
    #         K_nonzero @ Sv_detached.view(*Sv_batch_shape, I, 1, K, 1)
    #     ).squeeze(-1)
    #     StKS: Float[torch.Tensor, "... I I"] = torch.einsum("...ijk,...jk->...ij", StK_nonzero, Sv_detached)
    #     StKS = self.__class__.GetSvGradThroughStKS.apply(StKS, StK_nonzero.detach(), Sv)
    #     # ^^^ note this functions is an autograd hack to ensure that the gradients of Sv get populated on the backward pass
    #     # the forward pass is a no-op; it just returns StKS
    #     # on the backward pass, we set Sv's gradient to be what it would have been if we hadn't detached it a few lines above
    #     return StKS
