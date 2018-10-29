from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
