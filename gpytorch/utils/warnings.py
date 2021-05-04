#!/usr/bin/env python3


class GPInputWarning(UserWarning):
    """
    Warning thrown when a GP model receives an unexpected input.
    For example, when an :obj:`~gpytorch.models.ExactGP` in eval mode receives the training data as input.
    """

    pass


class NumericalWarning(RuntimeWarning):
    """
    Warning thrown when convergence criteria are not met, or when comptuations require extra stability.
    """

    pass


class OldVersionWarning(UserWarning):
    """
    Warning thrown when loading a saved model from an outdated version of GPyTorch.
    """

    pass
