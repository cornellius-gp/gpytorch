"""Abstract base class for Gaussian process approximation strategies."""

from __future__ import annotations

import abc

from typing import Optional, TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from ... import distributions, Module

if TYPE_CHECKING:
    from ... import kernels, likelihoods, means


# TODO: Should this inherit from Module? Probably, since we might want to autodiff through its parameters (e.g. actions)
class ApproximationStrategy(Module, abc.ABC):
    """Abstract base class for Gaussian process approximation strategies."""

    def __init__(self) -> None:
        super().__init__()

    def init_cache(
        self,
        mean: means.Mean,
        kernel: kernels.Kernel,
        likelihood: likelihoods._GaussianLikelihoodBase,
        train_inputs: Optional[Float[Tensor, "N D"]] = None,
        train_targets: Optional[Float[Tensor, " N"]] = None,
    ) -> None:
        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood
        self.train_inputs = train_inputs
        self.train_targets = train_targets

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

    def prior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the prior distribution of the Gaussian process at the given inputs."""
        # TODO: check whether this is a vector-valued / multitask GP here to use the right distribution?
        return distributions.MultivariateNormal(self.mean(inputs), self.kernel(inputs))

    @abc.abstractmethod
    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the approximate posterior distribution of the Gaussian process."""
        raise NotImplementedError
