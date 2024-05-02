#!/usr/bin/env python3
"""Computation-aware Gaussian processes."""

import math
import warnings
from typing import Optional

import torch
from linear_operator import operators, utils as linop_utils

from ... import kernels, settings
from ...distributions import MultivariateNormal
from ...utils.warnings import GPInputWarning
from ..exact_gp import ExactGP
from ..exact_prediction_strategies import ComputationAwarePredictionStrategy
from .linear_solvers import LinearSolver
from .preconditioners import Preconditioner


class ComputationAwareGP(ExactGP):
    """Computation-aware Gaussian process.

    :param train_inputs: (size n x d) The training features :math:`\mathbf X`.
    :param train_targets: (size n) The training targets :math:`\mathbf y`.
    :param likelihood: The Gaussian likelihood that defines the observational distribution.
    :param linear_solver: The (probabilistic) linear solver used for inference.
    :param preconditioner: The preconditioner for the covariance matrix :math:`K(X, X) + \Lambda`.

    Example:
        >>> TODO
    """

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood,
        preconditioner: Preconditioner,
        linear_solver: LinearSolver,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.preconditioner = preconditioner  # Registers preconditioner's parameters as hyperparameters
        self._linear_solver = linear_solver
        self._solver_state = None

        # Register policy hyperparameters
        for param_name, param in self._linear_solver.policy.named_parameters():
            self.register_parameter(name="linear_solver_policy_" + param_name, parameter=param)

    def preconditioner_augmented_forward(self, x):
        """Compute the mean and covariance function augmented by a preconditioner."""
        if self.preconditioner is None:
            return MultivariateNormal(self.mean_module(x), self.covar_module(x))

        X_train = self.train_inputs[0]
        y_train = self.train_targets

        # Preconditioner-augmented mean function
        mean_x = self.mean_module(x) + self.covar_module(x, X_train) @ (
            self.preconditioner.inv_matmul(
                y_train - self.mean_module(X_train),
                kernel=self.covar_module,
                noise=self.likelihood.noise,
                X=X_train,
            )
        )

        # Preconditioner-augmented covariance function

        # TODO: We can compute the diagonal more efficiently. Is this helpful here?
        # if diag:

        #     def kxX_Pinv_kXx(kernel_Xx):
        #         return kernel_Xx.T @ self.preconditioner.inv_matmul(kernel_Xx)

        #     return self.covar_module(x1, x2, diag=diag) - torch.vmap(kxX_Pinv_kXx)(
        #         self.covar_module(x1, X_train).to_dense()
        #     )

        # TODO: this should probably be a linear operator, such that the diagonal can be computed efficiently
        covar_x = self.covar_module(x) - self.covar_module(x, X_train) @ (
            self.preconditioner.inv_matmul(  # TODO: replace with square root => more numerically stable
                self.covar_module(X_train, x),
                kernel=self.covar_module,
                noise=self.likelihood.noise,
                X=X_train,
            )
        )
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimization
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            # if settings.debug.on():
            #     if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
            #         raise RuntimeError("You must train on the training inputs!")
            # NOTE: Not true for batched training.
            return self.preconditioner_augmented_forward(*inputs, **kwargs)

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = self.forward(
                *full_inputs, **kwargs
            )  # TODO: should this be the original prior or the preconditioner-augmented prior?
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ComputationAwareGP.forward must return a MultivariateNormal.")
            return full_output

        # Posterior mode
        else:
            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:

                # Evaluate preconditioner-augmented prior distribution
                train_preconditioner_augmented_prior_dist = self.preconditioner_augmented_forward(
                    *train_inputs, **kwargs
                )

                # Create the prediction strategy for
                self.prediction_strategy = ComputationAwarePredictionStrategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_preconditioner_augmented_prior_dist,
                    train_labels=self.train_targets,
                    kernel=self.covar_module,
                    likelihood=self.likelihood,
                    preconditioner=self.preconditioner,
                    linear_solver=self.linear_solver,
                    solver_state=self._solver_state,
                )  # TODO: ensure prediction strategy (and MLL) use preconditioner

            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training/test data
            # TODO: make sure this is the preconditioner-augmented prior / kernel
            full_output = self.preconditioner_augmented_forward(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ComputationAwareGP.forward must return a MultivariateNormal.")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            (
                predictive_mean,
                predictive_covar,
            ) = self.prediction_strategy.exact_prediction(
                full_mean, full_covar
            )  # TODO: replace with "preconditioned" prediction call

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)

    @property
    def linear_solver(self) -> LinearSolver:
        """Linear solver used to compute the posterior."""
        return self._linear_solver

    @linear_solver.setter
    def linear_solver(self, linear_solver: LinearSolver) -> None:
        self._linear_solver = linear_solver


class ComputationAwareGPOpt(ExactGP):
    """Computation-aware Gaussian process."""

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood,
        num_iter: int,
        num_non_zero: int,
        chunk_size: Optional[int] = None,
        initialization: str = "targets",
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.num_iter = num_iter
        self.num_non_zero = num_non_zero
        self.chunk_size = chunk_size

        num_train = train_targets.size(-1)

        # # Random initialization
        # non_zero_idcs = torch.cat(
        #     [
        #         torch.randperm(num_train, device=train_inputs.device)[: self.num_non_zero].unsqueeze(-2)
        #         for _ in range(self.num_iter)
        #     ],
        #     dim=0,
        # )
        # blocks = torch.nn.Parameter(
        #     torch.randn(
        #         self.num_iter,
        #         self.num_non_zero,
        #         dtype=train_inputs.dtype,
        #         device=train_inputs.device,
        #     )
        # )
        # Initialize with rhs of linear system (i.e. train targets)
        non_zero_idcs = torch.arange(
            (num_train // num_iter) * num_iter,
            # int(math.sqrt(num_train / num_iter)) * num_iter,  # TODO: This does not include all training datapoints!
            device=train_inputs.device,
        ).reshape(self.num_iter, -1)
        self.num_non_zero = non_zero_idcs.shape[1]

        if initialization == "random":
            self.non_zero_action_entries = torch.nn.Parameter(
                torch.randn_like(
                    non_zero_idcs,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                ).div(math.sqrt(self.num_non_zero))
            )
        elif initialization == "targets":
            self.non_zero_action_entries = torch.nn.Parameter(
                train_targets.clone()[: (num_train // num_iter) * num_iter].reshape(self.num_iter, -1)
                # train_targets.clone()[: int(math.sqrt(num_train / num_iter)) * num_iter].reshape(self.num_iter, -1)
            )
            self.non_zero_action_entries.div(
                torch.linalg.vector_norm(self.non_zero_action_entries, dim=1).reshape(-1, 1)
            )
        else:
            raise ValueError(f"Unknown initialization: '{initialization}'.")

        self.actions = (
            operators.BlockSparseLinearOperator(  # TODO: Can we speed this up by allowing ranges as non-zero indices?
                non_zero_idcs=non_zero_idcs, blocks=self.non_zero_action_entries, size_sparse_dim=num_train
            )
        )

    def __call__(self, x):
        if self.training:
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        else:
            # covar_train_train = self.covar_module(self.train_inputs[0])

            kernel = self.covar_module

            if isinstance(kernel, kernels.ScaleKernel):
                outputscale = kernel.outputscale
                lengthscale = kernel.base_kernel.lengthscale
                kernel_forward_fn = kernel.base_kernel._forward
            else:
                outputscale = 1.0
                lengthscale = kernel.lengthscale
                kernel_forward_fn = kernel._forward

            actions_op = self.actions

            gram_SKS = kernels.SparseQuadForm.apply(
                self.train_inputs[0] / lengthscale,
                actions_op.blocks,
                # actions_op.non_zero_idcs.mT,
                None,
                kernel_forward_fn,
                None,
                self.chunk_size,
            )

            StrS_diag = (actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
            gram_SKhatS = outputscale * gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)

            if x.ndim == 1:
                x = torch.atleast_2d(x).mT
            covar_x_train_actions = outputscale * kernels.SparseLinearForm.apply(
                x / lengthscale,
                self.train_inputs[0] / lengthscale,
                actions_op.blocks,
                actions_op.non_zero_idcs,
                kernel_forward_fn,
                None,
                self.chunk_size,  # TODO: the chunk size should probably be larger here, since we usually compute this on the test set
            )

            cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                gram_SKhatS.to(dtype=torch.float64), upper=False
            )
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # Compressed representer weights
            actions_target = self.actions @ (self.train_targets - self.mean_module(self.train_inputs[0]))
            compressed_repr_weights = (
                torch.cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x) - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)

            return MultivariateNormal(mean, covar)

    # def __call__(self, x):
    #     if self.training:
    #         return MultivariateNormal(
    #             self.mean_module(x),
    #             self.covar_module(x),
    #         )
    #     else:
    #         # covar_train_train = self.covar_module(self.train_inputs[0])

    #         kernel = self.covar_module

    #         with torch.no_grad():
    #             outputscale = 1.0
    #             lengthscale = 1.0

    #             if isinstance(kernel, kernels.ScaleKernel):
    #                 outputscale = kernel.outputscale
    #                 lengthscale = kernel.base_kernel.lengthscale
    #                 forward_fn = kernel.base_kernel._forward
    #             else:
    #                 lengthscale = kernel.lengthscale

    #                 forward_fn = kernel._forward

    #         actions_op = self.actions

    #         gram_SKS, StrS = kernels.SparseBilinearForms.apply(
    #             self.train_inputs[0] / lengthscale,
    #             actions_op.blocks.mT,
    #             actions_op.blocks.mT,
    #             actions_op.non_zero_idcs.mT,
    #             actions_op.non_zero_idcs.mT,
    #             forward_fn,
    #             None,
    #             self.chunk_size,
    #         )  # TODO: Can compute StrS more efficiently since we assume actions are made up of non-intersecting blocks.
    #         gram_SKhatS = outputscale * gram_SKS + self.likelihood.noise * StrS

    #         if x.ndim == 1:
    #             x = torch.atleast_2d(x).mT
    #         covar_x_train_actions = outputscale * kernels.SparseLinearForms.apply(
    #             x / lengthscale,
    #             self.train_inputs[0] / lengthscale,
    #             actions_op.blocks,
    #             actions_op.non_zero_idcs,
    #             forward_fn,
    #             None,
    #             self.chunk_size,  # TODO: the chunk size should probably be larger here, since we usually compute this on the test set
    #         )

    #         cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
    #             gram_SKhatS.to(dtype=torch.float64), upper=False
    #         )
    #         covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
    #             cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
    #         ).mT

    #         # Compressed representer weights
    #         actions_target = self.actions @ (self.train_targets - self.mean_module(self.train_inputs[0]))
    #         compressed_repr_weights = (
    #             torch.cholesky_solve(
    #                 actions_target.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
    #             )
    #             .squeeze(-1)
    #             .to(self.train_inputs[0].dtype)
    #         )

    #         mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
    #         covar = self.covar_module(x) - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)

    #         return MultivariateNormal(mean, covar)
