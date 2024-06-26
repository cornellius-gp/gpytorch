#!/usr/bin/env python3
"""Computation-aware iterative Gaussian processes."""

import warnings
from typing import Optional

import torch
from linear_operator.operators import BlockSparseLinearOperator, LinearOperator, RootLinearOperator, ZeroLinearOperator

from ... import settings
from ...distributions import MultivariateNormal
from ...utils.memoize import cached
from ...utils.warnings import GPInputWarning
from ..exact_gp import ExactGP
from ..exact_prediction_strategies import DefaultPredictionStrategy
from .linear_solvers import LinearSolver, LinearSolverState
from .preconditioners import Preconditioner


class ComputationAwarePredictionStrategy(DefaultPredictionStrategy):
    """A Gaussian process prediction strategy which considers the performed computation."""

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_prior_dist,
        train_labels: torch.Tensor,
        likelihood,
        kernel,
        preconditioner: Preconditioner,
        linear_solver: LinearSolver,
        solver_state: LinearSolverState,
    ):
        super().__init__(
            train_inputs,
            train_prior_dist,
            train_labels,
            likelihood,
            root=None,
            inv_root=None,
        )
        if solver_state is not None:
            if solver_state.cache["actions_op"].shape[1] == train_inputs[0].shape[0]:
                # Linear solver was run during loss computation, therefore we can recycle the solver state.
                self._solver_state = solver_state
                return None

        # We only want to predict using the posterior, nothing has been precomputed (e.g. if we evaluated the loss on a batch).

        # Evaluate preconditioner-augmented prior
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix

        # Compute the representer weights
        with torch.no_grad():  # Ensure gradients are not taken through the solve
            self._solver_state = linear_solver.solve(
                train_train_covar,
                self.train_labels - train_mean,
                # TODO: arguments below are only needed for sparse bilinear form
                # train_inputs=train_inputs[0],
                # kernel=kernel,
                # noise=likelihood.noise,
            )

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        raise NotImplementedError(
            "Fantasy observation updates not (yet) supported for computation-aware Gaussian processes."
        )

    @property
    def solver_state(self) -> LinearSolverState:
        """State of the linear solver solving for the representer weights."""
        return self._solver_state

    def exact_prediction(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]
        # For efficiency - we can make things more efficient
        if joint_covar.size(-1) <= settings.max_eager_kernel_size.value():
            test_covar = joint_covar[..., self.num_train :, :].to_dense()
            test_test_covar = test_covar[..., self.num_train :]
            test_train_covar = test_covar[..., : self.num_train]
        else:
            test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
            test_train_covar = joint_covar[..., self.num_train :, : self.num_train]

        # Precompute K(X_*, X) S
        covar_test_train_actions = (
            (self.solver_state.cache["actions_op"]._matmul(test_train_covar.mT)).mT
            if isinstance(self.solver_state.cache["actions_op"], BlockSparseLinearOperator)
            else test_train_covar @ self.solver_state.cache["actions_op"].mT
        )

        return (
            self.predictive_mean(test_mean, covar_test_train_actions),
            self.predictive_covar(test_test_covar, covar_test_train_actions),
        )

    @property
    @cached(name="mean_cache")
    def mean_cache(self) -> torch.Tensor:
        """Compute the representer weights."""
        raise NotImplementedError  # Note: Want to avoid quadratic cost of multiplying with representer weights.

    def predictive_mean(self, test_mean: torch.Tensor, covar_test_train_actions: torch.Tensor) -> torch.Tensor:
        """
        Computes the posterior predictive mean.

        :param Tensor test_mean: The test prior mean
        :param Tensor covar_test_train_actions: Kernel evaluated at test and train points, multiplied by the
        actions: :math:`K_{X_*, X} S`.

        :return: The predictive posterior mean of the test points
        """
        return test_mean + covar_test_train_actions @ self.solver_state.cache["compressed_solution"]

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        raise NotImplementedError

    def _exact_predictive_covar_inv_quad_form_root(self, covar_test_train_actions: torch.Tensor):
        r"""
        Computes :math:`K_{X_*, X} S L^{-T}`.

        :param Tensor covar_test_train_actions: Kernel evaluated at test and train points, multiplied by the
        actions: :math:`K_{X_*, X} S`.

        Returns
            :obj:`~linear_operator.operators.LinearOperator`: :math:`K_{X_*, X} S L^{-T}`
        """
        # Compute a triangular solve to obtain k(X_*, X)S L^{-T}
        return torch.linalg.solve_triangular(
            self.solver_state.cache["cholfac_gram"],
            covar_test_train_actions.mT,
            upper=False,
            left=True,
        ).mT

    def predictive_covar(self, test_test_covar: LinearOperator, test_train_covar: LinearOperator):
        """Computes the posterior predictive covariance."""
        if settings.skip_posterior_variances.on():
            return ZeroLinearOperator(*test_test_covar.size())

        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(test_train_covar)

        # TODO: mode for marginals only that vmaps over the test points to reduce memory complexity to O(n_test)
        return test_test_covar - RootLinearOperator(root=covar_inv_quad_form_root)


class ComputationAwareIterativeGP(ExactGP):
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
        linear_solver: LinearSolver,
        preconditioner: Optional[Preconditioner] = None,
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
                    raise RuntimeError("ComputationAwareIterativeGP.forward must return a MultivariateNormal.")
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
                    raise RuntimeError("ComputationAwareIterativeGP.forward must return a MultivariateNormal.")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            (predictive_mean, predictive_covar,) = self.prediction_strategy.exact_prediction(
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
