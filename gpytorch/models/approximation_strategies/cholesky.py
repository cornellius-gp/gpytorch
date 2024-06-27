"""Cholesky-based Gaussian process approximation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from .approximation_strategy import ApproximationStrategy

if TYPE_CHECKING:
    from ... import distributions


class Cholesky(ApproximationStrategy):
    def __init__(self) -> None:
        super().__init__()

    def posterior(self, inputs: Tensor) -> distributions.MultivariateNormal:

        # TODO: pull all these quantities from the cache instead and cache what you compute here
        return super().posterior(inputs)
