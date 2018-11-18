#!/usr/bin/env python3

from .gp import GP
from .exact_gp import ExactGP
from .variational_gp import VariationalGP
from .grid_inducing_variational_gp import GridInducingVariationalGP
from .additive_grid_inducing_variational_gp import AdditiveGridInducingVariationalGP
from .abstract_variational_gp import AbstractVariationalGP

try:
    from .pyro_variational_gp import PyroVariationalGP
except ImportError:

    class PyroVariationalGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroVariationalGP because you dont have Pyro installed.")


__all__ = [
    "GP",
    "ExactGP",
    "VariationalGP",
    "GridInducingVariationalGP",
    "AdditiveGridInducingVariationalGP",
    "AbstractVariationalGP",
    "PyroVariationalGP",
]
