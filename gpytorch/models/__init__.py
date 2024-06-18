#!/usr/bin/env python3

import warnings

from . import computation_aware_iterative_gp, deep_gps, exact_prediction_strategies, gplvm, pyro
from .approximate_gp import ApproximateGP
from .computation_aware_gp import ComputationAwareGP
from .computation_aware_iterative_gp import ComputationAwareIterativeGP
from .exact_gp import ExactGP
from .gp import GP
from .model_list import AbstractModelList, IndependentModelList
from .pyro import PyroGP

# Alternative name for ApproximateGP
VariationalGP = ApproximateGP

# Alternative name for ComputationAwareIterativeGP
IterGP = ComputationAwareIterativeGP

# Alternative name for ComputationAwareGP
CAGP = ComputationAwareGP


# Deprecated for 0.4 release
class AbstractVariationalGP(ApproximateGP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractVariationalGP has been renamed to ApproximateGP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


# Deprecated for 0.4 release
class PyroVariationalGP(ApproximateGP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("PyroVariationalGP has been renamed to PyroGP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


__all__ = [
    "AbstractModelList",
    "ApproximateGP",
    "ComputationAwareIterativeGP",
    "ComputationAwareGP",
    "CAGP",
    "IterGP",
    "ExactGP",
    "GP",
    "IndependentModelList",
    "PyroGP",
    "VariationalGP",
    "computation_aware_iterative_gp",
    "deep_gps",
    "gplvm",
    "exact_prediction_strategies",
    "pyro",
]
