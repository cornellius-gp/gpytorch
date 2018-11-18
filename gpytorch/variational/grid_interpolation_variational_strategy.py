#!/usr/bin/env python3

import torch
from ..utils.interpolation import Interpolation, left_interp
from ..lazy import InterpolatedLazyTensor
from ..distributions import MultivariateNormal
from .variational_strategy import VariationalStrategy


class GridInterpolationVariationalStrategy(VariationalStrategy):
    def __init__(self, model, grid_size, grid_bounds, variational_distribution):
        grid = torch.zeros(grid_size, len(grid_bounds))
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size)

        inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        prev_points = None
        for i in range(len(grid_bounds)):
            for j in range(grid_size):
                inducing_points[j * grid_size ** i : (j + 1) * grid_size ** i, i].fill_(grid[j, i])
                if prev_points is not None:
                    inducing_points[j * grid_size ** i : (j + 1) * grid_size ** i, :i].copy_(prev_points)
            prev_points = inducing_points[: grid_size ** (i + 1), : (i + 1)]

        super(GridInterpolationVariationalStrategy, self).__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        object.__setattr__(self, "model", model)

        self.register_buffer("grid", grid)

    def _compute_grid(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs)
        return interp_indices, interp_values

    def forward(self, x):
        variational_distribution = self.variational_distribution.variational_distribution

        # Get interpolations
        interp_indices, interp_values = self._compute_grid(x)

        # Compute test mean
        # Left multiply samples by interpolation matrix
        predictive_mean = left_interp(interp_indices, interp_values, variational_distribution.mean.unsqueeze(-1))
        predictive_mean = predictive_mean.squeeze(-1)

        # Compute test covar
        predictive_covar = InterpolatedLazyTensor(
            variational_distribution.lazy_covariance_matrix,
            interp_indices,
            interp_values,
            interp_indices,
            interp_values,
        )

        output = MultivariateNormal(predictive_mean, predictive_covar)
        return output
