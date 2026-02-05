#!/usr/bin/env python3

from __future__ import annotations

from linear_operator.utils.warnings import NumericalWarning


class GPInputWarning(UserWarning):
    """
    Warning thrown when a GP model receives an unexpected input.
    For example, when an :obj:`~gpytorch.models.ExactGP` in eval mode receives the training data as input.
    """


class OldVersionWarning(UserWarning):
    """
    Warning thrown when loading a saved model from an outdated version of GPyTorch.
    """


__all__ = [
    "GPInputWarning",
    "OldVersionWarning",
    "NumericalWarning",
]
