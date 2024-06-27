"""Approximation strategies for Gaussian processes."""

from .approximation_strategy import ApproximationStrategy
from .cholesky import Cholesky

__all__ = [
    "ApproximationStrategy",
    "Cholesky",
]
