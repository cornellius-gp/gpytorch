"""Gaussian process model."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from .. import likelihoods, settings
from ..module import Module
from . import approximation_strategies

if TYPE_CHECKING:
    from .. import distributions, kernels, means


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
        likelihood: likelihoods._GaussianLikelihoodBase = likelihoods.GaussianLikelihood(),
        train_inputs: Optional[Float[Tensor, "N D"]] = None,
        train_targets: Optional[Float[Tensor, " N"]] = None,
        approximation_strategy: Optional[approximation_strategies.ApproximationStrategy] = None,
    ):
        # Input checking
        if not isinstance(likelihood, likelihoods._GaussianLikelihoodBase):
            raise TypeError(f"{self.__class__.__name__} only accepts Gaussian likelihoods.")

        super().__init__()

        self._mean = mean
        self._kernel = kernel
        self._likelihood = likelihood
        if approximation_strategy is not None:
            self.approximation_strategy = approximation_strategy
        elif (train_inputs is None) or (train_targets is None):
            self.approximation_strategy = approximation_strategies.Cholesky()
        elif train_inputs.shape[-1] <= settings.max_cholesky_size.value():
            self.approximation_strategy = approximation_strategies.Cholesky()
        else:
            # TODO: Choose a default approximation strategy here when not using Cholesky
            raise NotImplementedError

        self.approximation_strategy.init_cache(
            mean=self.mean,
            kernel=self.kernel,
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=self.likelihood,
        )
        # TODO: initialize approximation strategy by passing mean, kernel, likelihood and data and initialize its cache?
        # TODO: or: just initialize cache here via approximation_strategy.__class__.Cache(mean=...) and
        #  just call approximation_strategy.some_fn(args, cache), which reads from and writes to the cache

    @property
    def mean(self) -> means.Mean:
        return self._mean

    @mean.setter
    def mean(self, value: means.Mean):
        raise AttributeError("Cannot set mean of the GP after instantiation. Create a new model instead.")

    @property
    def kernel(self) -> means.Mean:
        return self._kernel

    @kernel.setter
    def kernel(self, value: kernels.Kernel):
        raise AttributeError("Cannot set kernel of the GP after instantiation. Create a new model instead.")

    @property
    def likelihood(self) -> means.Mean:
        return self._likelihood

    @likelihood.setter
    def likelihood(self, value: likelihoods._GaussianLikelihoodBase):
        raise AttributeError("Cannot set likelihood of the GP after instantiation. Create a new model instead.")

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

    def forward(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        inputs = inputs.unsqueeze(-1) if inputs.ndimension() <= 1 else inputs

        return self.approximation_strategy.prior(inputs)

    def prior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the prior distribution of the Gaussian process at the given inputs."""
        # This is just a more familiar interface for .forward
        return self.approximation_strategy.prior(inputs)

    def __call__(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        inputs = inputs.unsqueeze(-1) if inputs.ndimension() <= 1 else inputs

        # Training mode (Model selection / Hyperparameter optimization)
        if self.training:
            if self.train_inputs is None or self.train_targets is None:
                raise RuntimeError(
                    "Training inputs or targets cannot be 'None' in training mode. "
                    "Call my_model.eval() for prior predictions, or add training data "
                    "via my_model.train_inputs = ..., my_model.train_targets = ..."
                )

            return self.forward(inputs)
        else:
            # Evaluation / posterior mode
            return self.approximation_strategy.posterior(inputs)

    def predictive(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the posterior predictive distribution of the Gaussian process at the given inputs."""
        return self.likelihood(self.__call__(inputs))
