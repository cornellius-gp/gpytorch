#!/usr/bin/env python3


import torch

from ..distributions import Delta, MultivariateNormal
from ..variational.grid_interpolation_variational_strategy import GridInterpolationVariationalStrategy


class AdditiveGridInterpolationVariationalStrategy(GridInterpolationVariationalStrategy):
    def __init__(
        self, model, grid_size, grid_bounds, num_dim, variational_distribution, mixing_params=False, sum_output=True
    ):
        super(AdditiveGridInterpolationVariationalStrategy, self).__init__(
            model, grid_size, grid_bounds, variational_distribution
        )
        self.num_dim = num_dim
        self.sum_output = sum_output
        # Mixing parameters
        if mixing_params:
            self.register_parameter(name="mixing_params", parameter=torch.nn.Parameter(torch.ones(num_dim) / num_dim))

    @property
    def prior_distribution(self):
        """
        If desired, models can compare the input to forward to inducing_points and use a GridKernel for space
        efficiency.

        However, when using a default VariationalDistribution which has an O(m^2) space complexity anyways, we find that
        GridKernel is typically not worth it due to the moderate slow down of using FFTs.
        """
        out = super(AdditiveGridInterpolationVariationalStrategy, self).prior_distribution
        mean = out.mean.repeat(self.num_dim, 1)
        covar = out.lazy_covariance_matrix.repeat(self.num_dim, 1, 1)
        return MultivariateNormal(mean, covar)

    def _compute_grid(self, inputs):
        num_data, num_dim = inputs.size()
        inputs = inputs.transpose(0, 1).reshape(-1, 1)
        interp_indices, interp_values = super(AdditiveGridInterpolationVariationalStrategy, self)._compute_grid(inputs)
        interp_indices = interp_indices.view(num_dim, num_data, -1)
        interp_values = interp_values.view(num_dim, num_data, -1)

        if hasattr(self, "mixing_params"):
            interp_values = interp_values.mul(self.mixing_params.unsqueeze(1).unsqueeze(2))
        return interp_indices, interp_values

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        if x.ndimension() == 1:
            x = x.unsqueeze(-1)
        elif x.ndimension() != 2:
            raise RuntimeError("AdditiveGridInterpolationVariationalStrategy expects a 2d tensor.")

        num_data, num_dim = x.size()
        if num_dim != self.num_dim:
            raise RuntimeError("The number of dims should match the number specified.")

        output = super().forward(x, inducing_points, inducing_values, variational_inducing_covar)
        if self.sum_output:
            if variational_inducing_covar is not None:
                mean = output.mean.sum(0)
                covar = output.lazy_covariance_matrix.sum(-3)
                return MultivariateNormal(mean, covar)
            else:
                return Delta(output.mean.sum(0))
        else:
            return output
