#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import deque
from typing import Tuple, Union

import torch
from linear_operator import operators

from ..likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from .marginal_log_likelihood import MarginalLogLikelihood


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
        solver_state = self.model.linear_solver.solve(Khat, target)
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
        linear_op_actions = (
            actions_op._matmul(Khat).mT
            if isinstance(actions_op, operators.BlockSparseLinearOperator)
            else Khat @ actions_op.mT
        )

    with torch.set_grad_enabled(True):
        # Explicit gradient with S instead of dS/dparams that is linear in S
        gradient_helper = -torch.sum(  # TODO: should there be a minus here?
            (
                torch.outer(residual, compressed_repr_weights)
                + torch.cholesky_solve(linear_op_actions.mT, cholfac_gram, upper=False).mT
            )
            * actions_op.mT
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
            actions_op._matmul(
                (actions_op._matmul(lin_op_copy._linear_op) + actions_op._matmul(lin_op_copy._diag_tensor)).mT
            )  # Workaround for gradient bug when using BlockSparseLinearOperator
            # TODO: we can do better here by pulling the diagonal matrix out: S'diag S, which avoids extra O(ni) memory
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
