#!/usr/bin/env python3
"""Computation-aware Gaussian processes."""

import warnings

import torch

from ... import settings
from ...distributions import MultivariateNormal
from ...utils.warnings import GPInputWarning
from ..exact_gp import ExactGP
from ..exact_prediction_strategies import ComputationAwarePredictionStrategy
from .linear_solvers import LinearSolver


class ComputationAwareGP(ExactGP):
    """Computation-aware Gaussian process.

    :param train_inputs: (size n x d) The training features :math:`\mathbf X`.
    :param train_targets: (size n) The training targets :math:`\mathbf y`.
    :param likelihood: The Gaussian likelihood that defines the observational distribution.
    :param linear_solver: The (probabilistic) linear solver used for inference.

    Example:
        >>> TODO
    """

    def __init__(self, train_inputs: torch.Tensor, train_targets, likelihood, linear_solver: LinearSolver):
        super().__init__(train_inputs, train_targets, likelihood)
        self._linear_solver = linear_solver
        self._solver_state = None

        # Register policy hyperparameters
        for param_name, param in self._linear_solver.policy.named_parameters():
            self.register_parameter(name="linear_solver_policy_" + param_name, parameter=param)

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
            if settings.debug.on():
                if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            res = self.forward(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = self.forward(*full_inputs, **kwargs)
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
                train_output = self.forward(*train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = ComputationAwarePredictionStrategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    kernel=self.covar_module,
                    likelihood=self.likelihood,
                    linear_solver=self.linear_solver,
                    solver_state=self._solver_state,
                )

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
            full_output = self.forward(*full_inputs, **kwargs)
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
            ) = self.prediction_strategy.exact_prediction(full_mean, full_covar)

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
