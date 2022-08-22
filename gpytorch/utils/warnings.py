#!/usr/bin/env python3

from linear_operator.utils.warnings import NumericalWarning


class GPInputWarning(UserWarning):
    """
    Warning thrown when a GP model receives an unexpected input.
    For example, when an :obj:`~gpytorch.models.ExactGP` in eval mode receives the training data as input.
    """

    pass


class OldVersionWarning(UserWarning):
    """
    Warning thrown when loading a saved model from an outdated version of GPyTorch.
    """

    pass


__all__ = [
    "GPInputWarning",
    "OldVersionWarning",
    "NumericalWarning",
]
