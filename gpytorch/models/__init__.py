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
    from .generic_variational_particle_gp import GenericVariationalParticleGP
    from .generic_variational_gaussian_gp import GenericVariationalGaussianGP

except ImportError as e:
    print(e)

    class PyroVariationalGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroVariationalGP because you dont have Pyro installed.")


    class GenericVariationalParticleGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a GenericVariationalParticleGP because you dont have Pyro installed.")


    class GenericVariationalGaussianGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a GenericVariationalGaussianGP because you dont have Pyro installed.")


__all__ = [
    "AbstractModelList",
    "AbstractVariationalGP",
    "AdditiveGridInducingVariationalGP",
    "ExactGP",
    "GenericVariationalGaussianGP",
    "GenericVariationalParticleGP",
    "GP",
    "GridInducingVariationalGP",
    "IndependentModelList",
    "PyroVariationalGP",
    "VariationalGP",
    "deep_gps",
]
