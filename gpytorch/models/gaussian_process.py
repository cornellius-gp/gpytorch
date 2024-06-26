"""Gaussian process model."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from .. import approximation_strategies, likelihoods, settings

from ..module import Module

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
        self._train_inputs = train_inputs
        self._train_targets = train_targets
        self.likelihood = likelihood
        if self.approximation_strategy is not None:
            self.approximation_strategy = approximation_strategy
        elif self.train_inputs.shape[-1] <= settings.max_cholesky_size.value():
            self.approximation_strategy = approximation_strategies.Cholesky()
        else:
            raise NotImplementedError
            # TODO: Choose a default approximation strategy here when not using Cholesky

    @property
    def train_inputs(self):
        return self._train_inputs

    @train_inputs.setter
    def train_inputs(self, value):
        self._train_inputs = value

    @property
    def train_targets(self):
        return self._train_targets

    @train_targets.setter
    def train_targets(self, value):
        self._train_targets = value

    def prior(self, inputs: Float[Tensor, "M D"]):
        pass

    def __call__(self, inputs: Float[Tensor, "M D"]):

        # Training mode (Model selection / Hyperparameter optimization)

        # Evaluation / posterior mode

        pass
