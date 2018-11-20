#!/usr/bin/env python3

from ..variational import CholeskyVariationalDistribution, AdditiveGridInterpolationVariationalStrategy
from ..models.abstract_variational_gp import AbstractVariationalGP
import warnings


class AdditiveGridInducingVariationalGP(AbstractVariationalGP):
    def __init__(self, grid_size, grid_bounds, num_dim, mixing_params=False, sum_output=True):
        warnings.warn(
            "AdditiveGridInducingVariationalGP is deprecated in favor of a new variational inference interface, "
            "and will be removed in a future release. Please see the new examples.",
            DeprecationWarning,
        )
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=grid_size, batch_size=num_dim)
        variational_strategy = AdditiveGridInterpolationVariationalStrategy(
            self,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
            num_dim=num_dim,
            variational_distribution=variational_distribution,
            mixing_params=mixing_params,
            sum_output=sum_output,
        )
        super(AdditiveGridInducingVariationalGP, self).__init__(variational_strategy)
