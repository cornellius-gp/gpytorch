#!/usr/bin/env python3

from ..variational import CholeskyVariationalDistribution, GridInterpolationVariationalStrategy
from .abstract_variational_gp import AbstractVariationalGP
import warnings


class GridInducingVariationalGP(AbstractVariationalGP):
    def __init__(self, grid_size, grid_bounds):
        warnings.warn(
            "GridInducingVariationalGP is deprecated in favor of a new variational inference interface, and will "
            "be removed in a future release. Please see the new examples.",
            DeprecationWarning,
        )
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=int(pow(grid_size, len(grid_bounds)))
        )
        variational_strategy = GridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=grid_bounds, variational_distribution=variational_distribution
        )
        super(GridInducingVariationalGP, self).__init__(variational_strategy)
