#!/usr/bin/env python3

from .gp import GP
from .abstract_variational_gp import AbstractVariationalGP
from .additive_grid_inducing_variational_gp import AdditiveGridInducingVariationalGP
from .exact_gp import ExactGP
from .grid_inducing_variational_gp import GridInducingVariationalGP
from .model_list import AbstractModelList, IndependentModelList
from .variational_gp import VariationalGP
from . import deep_gps


try:
    from .pyro_variational_gp import PyroVariationalGP
except ImportError:

    class PyroVariationalGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroVariationalGP because you dont have Pyro installed.")


__all__ = [
    "AbstractModelList",
    "AbstractVariationalGP",
    "AdditiveGridInducingVariationalGP",
    "ExactGP",
    "GP",
    "GridInducingVariationalGP",
    "IndependentModelList",
    "PyroVariationalGP",
    "VariationalGP",
    "deep_gps",
]
