"""Gaussian process model with computations performed via a Cholesky decomposition."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from ... import distributions, likelihoods

from .. import approximation_strategies

from ..gaussian_process import GaussianProcess

if TYPE_CHECKING:
    from ... import kernels, means


class CholeskyGP(GaussianProcess):
    """Gaussian process model with computations performed via a Cholesky decomposition.

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
    >>> model = models.zoo.CholeskyGP(
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
    ):
        super().__init__(
            likelihood=likelihood,
            approximation_strategy=approximation_strategies.Cholesky(),
            train_inputs=train_inputs,
            train_targets=train_targets,
        )
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        inputs = inputs.unsqueeze(-1) if inputs.ndimension() <= 1 else inputs

        return distributions.MultivariateNormal(self.mean_module(inputs), self.covar_module(inputs))
