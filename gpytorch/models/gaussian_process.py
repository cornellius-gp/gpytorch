"""Gaussian process model."""

from __future__ import annotations

import abc

from typing import Optional, TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from .. import likelihoods, settings

from ..module import Module

if TYPE_CHECKING:
    from .. import distributions
    from . import approximation_strategies


class GaussianProcess(Module, abc.ABC):
    """Base class for Gaussian process models.

    :param mean: Prior mean function.
    :param kernel: Prior kernel / covariance function.
    :param train_inputs: Training inputs :math:`\mathbf X`.
    :param train_targets: Training targets :math:`\mathbf y`.
    :param likelihood: Gaussian likelihood defining the observational noise distribution.
    :param approximation_strategy: Defines how to approximate costly computations necessary for large-scale datasets.

    Example:
    >>> from gpytorch import models, means, kernels, likelihoods
    >>>
    >>> # Prepare dataset
    >>> # train_x = ...
    >>> # train_y = ...
    >>> # test_x = ...
    >>>
    >>> # Define Gaussian process model
    >>> class MyGP(gpytorch.models.GaussianProcess):
    >>>     def __init__(self, train_inputs, train_targets, likelihood):
    >>>         super().__init__(train_inputs, train_targets, likelihood)
    >>>         self.mean_module = gpytorch.means.ZeroMean()
    >>>         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    >>>
    >>>     def forward(self, x):
    >>>         mean = self.mean_module(x)
    >>>         covar = self.covar_module(x)
    >>>         return gpytorch.distributions.MultivariateNormal(mean, covar)
    >>>
    >>> model = MyGP(
    ...     train_inputs=train_x,
    ...     train_targets=train_y,
    ...     likelihood=likelihoods.GaussianLikelihood(),
    ... )
    >>>
    >>> # GP posterior for the latent function
    >>> model(test_x)
    >>>
    >>> # Posterior predictive distribution for the observations
    >>> model.predictive(test_x)    # Equivalent to model.likelihood(model(test_x))

    """

    def __init__(
        self,
        likelihood: likelihoods._GaussianLikelihoodBase,
        train_inputs: Optional[Float[Tensor, "N D"]] = None,
        train_targets: Optional[Float[Tensor, " N"]] = None,
        approximation_strategy: Optional[approximation_strategies.ApproximationStrategy] = None,
    ):
        # Input checking
        if not isinstance(likelihood, likelihoods._GaussianLikelihoodBase):
            raise TypeError(f"{self.__class__.__name__} only accepts Gaussian likelihoods.")

        super().__init__()

        self.likelihood = likelihood

        # TODO: Do we want to choose defaults here or do we want to force the user to make a decision when subclassing?
        if approximation_strategy is not None:
            self.approximation_strategy = approximation_strategy
        elif (train_inputs is None) or (train_targets is None):
            self.approximation_strategy = approximation_strategies.Cholesky()
        elif train_inputs.shape[-1] <= settings.max_cholesky_size.value():
            self.approximation_strategy = approximation_strategies.Cholesky()
        else:
            # TODO: Choose a default approximation strategy here when not using Cholesky
            raise NotImplementedError

        # Do not allow instantiation of a GP without specifying an approximation strategy.
        assert self.approximation_strategy is not None, "Trying to instantiate a GP without an ApproximationStrategy."

        self.approximation_strategy.init_cache(
            model=self,  # NOTE: Introduces circular reference.
            train_inputs=train_inputs,
            train_targets=train_targets,
        )

    @property
    def train_inputs(self) -> Float[Tensor, "N D"]:
        return self.approximation_strategy.train_inputs

    @train_inputs.setter
    def train_inputs(self, value: Optional[Float[Tensor, "N D"]]):
        self.approximation_strategy.train_inputs = value

    @property
    def train_targets(self) -> Optional[Float[Tensor, " N"]]:
        return self.approximation_strategy.train_targets

    @train_targets.setter
    def train_targets(self, value: Optional[Float[Tensor, " N"]]):
        self.approximation_strategy.train_targets = value

    def eval(self):
        self.likelihood.eval()
        return super().eval()

    def train(self, mode=True):
        self.likelihood.train(mode)
        return super().train(mode)

    @abc.abstractmethod
    def forward(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        raise NotImplementedError()

    def prior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the prior distribution of the Gaussian process at the given inputs."""
        # This is just a more familiar interface for .forward
        return self.forward(inputs)

    def __call__(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        inputs = inputs.unsqueeze(-1) if inputs.ndimension() <= 1 else inputs

        if self.training:
            # Training mode (Model selection / Hyperparameter optimization)
            if self.train_inputs is None or self.train_targets is None:
                raise RuntimeError(
                    "Training inputs or targets cannot be 'None' in training mode. "
                    "Call my_model.eval() for prior predictions, or add training data "
                    "via my_model.train_inputs = ..., my_model.train_targets = ..."
                )

            return self.forward(inputs)
        elif settings.prior_mode.on() or (self.train_inputs is None) or (self.train_targets is None):
            # Prior mode
            return self.forward(inputs)
        else:
            # Posterior / evaluation mode
            return self.approximation_strategy.posterior(inputs)

    def predictive(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the posterior predictive distribution of the Gaussian process at the given inputs."""
        return self.likelihood(self.__call__(inputs))
