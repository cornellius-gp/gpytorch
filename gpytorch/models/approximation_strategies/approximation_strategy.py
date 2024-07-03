"""Abstract base class for Gaussian process approximation strategies."""

from __future__ import annotations

import abc

from typing import Optional

from jaxtyping import Float
from torch import Tensor

from ... import distributions, Module


class ApproximationStrategy(abc.ABC, Module):
    """Abstract base class for Gaussian process approximation strategies."""

    def __init__(self) -> None:
        super().__init__()

    def init_cache(
        self,
        model: Module,
        train_inputs: Optional[Float[Tensor, "N D"]] = None,
        train_targets: Optional[Float[Tensor, " N"]] = None,
    ) -> None:

        # Set model as an attribute of the ApproximationStrategy without registering it as a
        # submodule of ApproximationStrategy by bypassing Module.__setattr__ explicitly.
        object.__setattr__(self, "model", model)

        self.train_inputs = train_inputs
        self.train_targets = train_targets

    @property
    def train_inputs(self) -> Float[Tensor, "N D"]:
        return self._train_inputs

    @train_inputs.setter
    def train_inputs(self, value: Optional[Float[Tensor, "N D"]]):
        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        # TODO: release cache here
        if value is None:
            self._train_inputs = value
        else:
            self._train_inputs = value.unsqueeze(-1) if value.ndimension() <= 1 else value

    @property
    def train_targets(self) -> Optional[Float[Tensor, " N"]]:
        return self._train_targets

    @train_targets.setter
    def train_targets(self, value: Optional[Float[Tensor, " N"]]):
        # TODO: release cache here
        self._train_targets = value

    def prior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the prior distribution of the Gaussian process at the given inputs."""
        # TODO: check whether this is a vector-valued / multitask GP here to use the right distribution?
        return distributions.MultivariateNormal(self.model.mean(inputs), self.model.kernel(inputs))

    @abc.abstractmethod
    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the approximate posterior distribution of the Gaussian process."""
        raise NotImplementedError
