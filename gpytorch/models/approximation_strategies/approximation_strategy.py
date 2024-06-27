"""Abstract base class for Gaussian process approximation strategies."""

from __future__ import annotations

import abc

from typing import TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from ... import Module

if TYPE_CHECKING:
    from ... import distributions


# TODO: Should this inherit from Module? Probably, since we might want to autodiff through its parameters (e.g. actions)
class ApproximationStrategy(Module, abc.ABC):
    """Abstract base class for Gaussian process approximation strategies."""

    def __init__(self) -> None:
        super().__init__()

    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the approximate posterior distribution of the Gaussian process."""
        raise NotImplementedError
