#!/usr/bin/env python3

import warnings
from .gp import GP
from .approximate_gp import ApproximateGP
from .exact_gp import ExactGP
from .model_list import AbstractModelList, IndependentModelList
from .pyro import PyroGP
from . import deep_gps
from . import pyro


# Alternative name for ApproximateGP
VariationalGP = ApproximateGP


# Deprecated for 0.4 release
class AbstractVariationalGP(ApproximateGP):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AbstractVariationalGP has been renamed to ApproximateGP.",
            DeprecationWarning
        )
        super().__init__(*args, **kwargs)


# Deprecated for 0.4 release
class PyroVariationalGP(ApproximateGP):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PyroVariationalGP has been renamed to PyroGP.",
            DeprecationWarning
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "AbstractModelList",
    "ApproximateGP",
    "ExactGP",
    "GP",
    "IndependentModelList",
    "PyroGP",
    "VariationalGP",
    "deep_gps",
    "pyro",
]
