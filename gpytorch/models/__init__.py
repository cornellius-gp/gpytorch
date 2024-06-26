"""Gaussian process models."""

from . import deep_gps, exact_prediction_strategies, gplvm, pyro
from .approximate_gp import ApproximateGP
from .exact_gp import ExactGP
from .gaussian_process import GaussianProcess
from .model_list import AbstractModelList, IndependentModelList
from .pyro import PyroGP

# Aliases
GP = GaussianProcess
VariationalGP = ApproximateGP


__all__ = [
    "AbstractModelList",
    "ApproximateGP",
    "ExactGP",
    "GaussianProcess",
    "GP",
    "IndependentModelList",
    "PyroGP",
    "VariationalGP",
    "deep_gps",
    "gplvm",
    "exact_prediction_strategies",
    "pyro",
]
