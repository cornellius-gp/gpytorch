"""Gaussian process model."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from .. import distributions, likelihoods, settings
from ..module import Module
from . import approximation_strategies

if TYPE_CHECKING:
    from .. import kernels, means


class GaussianProcess(Module):
    """Gaussian process model.

    :param mean: Prior mean function.
    :param kernel: Prior kernel / covariance function.
    :param train_inputs: Training inputs :math:`\mathbf X`.
    :param train_targets: Training targets :math:`\mathbf y`.
    :param likelihood: Gaussian likelihood defining the observational noise distribution.
    :param approximation_strategy: Defines how to approximate costly computations necessary for large-scale datasets.

    Example:
    >>> from gpytorch import models, means, kernels
    >>>
    >>> # Prepare dataset
    >>> # train_x = ...
    >>> # train_y = ...
    >>> # test_x = ...
    >>>
    >>> # Define model
    >>> model = models.GaussianProcess(
    ...     means.ZeroMean(),
    ...     kernels.MaternKernel(nu=2.5),
    ...     train_inputs=train_x,
    ...     train_targets=train_y,
    ... )
    >>>
    >>> # GP posterior for the latent function
    >>> model(test_x)
    >>>
    >>> # Posterior predictive distribution for the observations
    >>> model.predictive(test_x)

    """

    def __init__(
        self,
        mean: means.Mean,
        kernel: kernels.Kernel,
        train_inputs: Optional[Float[Tensor, "N D"]] = None,
        train_targets: Optional[Float[Tensor, " N"]] = None,
        likelihood: likelihoods._GaussianLikelihoodBase = likelihoods.GaussianLikelihood(),
        approximation_strategy: Optional[approximation_strategies.ApproximationStrategy] = None,
    ):
        # Input checking
        if not isinstance(likelihood, likelihoods._GaussianLikelihoodBase):
            raise TypeError(f"{self.__class__.__name__} only accepts Gaussian likelihoods.")

        super().__init__()

        self.mean = mean
        self.kernel = kernel
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.likelihood = likelihood  # TODO: is it a problem that this is its own Module?
        if approximation_strategy is not None or (self.train_inputs is None and self.train_targets is None):
            self.approximation_strategy = approximation_strategy
        elif self.train_inputs.shape[-1] <= settings.max_cholesky_size.value():
            self.approximation_strategy = approximation_strategies.Cholesky()
        else:
            # TODO: Choose a default approximation strategy here when not using Cholesky
            raise NotImplementedError

        # TODO: initialize approximation strategy by passing mean, kernel, likelihood and data and initialize its cache?
        # TODO: or: just initialize cache here via self.approximation_strategy.__class__.Cache(mean=...)

    @property
    def train_inputs(self) -> Float[Tensor, "N D"]:
        return self._train_inputs

    @train_inputs.setter
    def train_inputs(self, value: Optional[Float[Tensor, "N D"]]):
        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        if value is None:
            self._train_inputs = value
        else:
            self._train_inputs = value.unsqueeze(-1) if value.ndimension() <= 1 else value

    @property
    def train_targets(self) -> Optional[Float[Tensor, " N"]]:
        return self._train_targets

    @train_targets.setter
    def train_targets(self, value: Optional[Float[Tensor, " N"]]):
        self._train_targets = value

    def eval(self):
        self.likelihood.eval()
        return super().eval()

    def train(self, mode=True):
        self.likelihood.train()
        return super().train(mode)

    def forward(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        inputs = inputs.unsqueeze(-1) if inputs.ndimension() <= 1 else inputs

        # TODO: check whether this is a vector-valued / multitask GP here to use the right distribution?
        return distributions.MultivariateNormal(self.mean(inputs), self.kernel(inputs))

    def prior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the prior distribution of the Gaussian process at the given inputs."""
        return self.forward(inputs)

    def __call__(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        inputs = inputs.unsqueeze(-1) if inputs.ndimension() <= 1 else inputs

        # Training mode (Model selection / Hyperparameter optimization)
        if self.training:
            if self.train_inputs is None or self.train_targets is None:
                raise RuntimeError(
                    "Training inputs or targets cannot be 'None' in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )

            return self.forward(inputs)
        else:
            # Evaluation / posterior mode
            return self.approximation_strategy.posterior(inputs)

    def predictive(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the posterior predictive distribution of the Gaussian process at the given inputs."""
        return self.likelihood(self.__call__(inputs))
