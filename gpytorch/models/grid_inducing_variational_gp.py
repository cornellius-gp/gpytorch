from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor, InterpolatedLazyTensor
from ..variational import MVNVariationalStrategy
from ..kernels.kernel import Kernel
from ..kernels.grid_kernel import GridKernel
from ..utils.interpolation import Interpolation, left_interp
from .. import beta_features
from .abstract_variational_gp import AbstractVariationalGP


class GridInducingVariationalGP(AbstractVariationalGP):
    def __init__(self, grid_size, grid_bounds):
        self._grid_mode = True
        self._kernels = set()

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

        super(GridInducingVariationalGP, self).__init__(inducing_points)
        self.register_buffer("grid", grid)

    def _compute_grid(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs)
        return interp_indices, interp_values

    def _initalize_variational_parameters(self, prior_output):
        mean_init = prior_output.mean.detach()
        mean_init += torch.randn_like(mean_init).mul_(1e-1)
        chol_covar_init = torch.eye(len(mean_init), dtype=mean_init.dtype, device=mean_init.device)
        chol_covar_init += torch.randn_like(chol_covar_init).mul_(1e-1)
        self.variational_mean.data.copy_(mean_init)
        self.chol_variational_covar.data.copy_(chol_covar_init)

    def covar_diag(self, inputs):
        grid_mode = self._grid_mode
        self._grid_mode = False
        res = super(GridInducingVariationalGP, self).covar_diag(inputs)
        self._grid_mode = grid_mode
        return res

    def __call__(self, inputs, **kwargs):
        # Prior output
        if self.training or beta_features.diagonal_correction.on():
            prior_output = self.prior_output()
            if not self.variational_params_initialized.item():
                self._initalize_variational_parameters(prior_output)
                self.variational_params_initialized.fill_(1)

        # Variational output
        variational_output = self.variational_output()

        # Update the variational distribution
        if self.training:
            # Initialize variational parameters, if necessary
            new_variational_strategy = MVNVariationalStrategy(variational_output, prior_output)
            self.update_variational_strategy("inducing_point_strategy", new_variational_strategy)

        # Get interpolations
        interp_indices, interp_values = self._compute_grid(inputs)

        # Compute test mean
        # Left multiply samples by interpolation matrix
        test_mean = left_interp(interp_indices, interp_values, variational_output.mean.unsqueeze(-1))
        test_mean = test_mean.squeeze(-1)

        # Compute test covar
        test_covar = InterpolatedLazyTensor(
            variational_output.lazy_covariance_matrix, interp_indices, interp_values, interp_indices, interp_values
        )

        # Diagonal correction
        if beta_features.diagonal_correction.on():
            from ..lazy import AddedDiagLazyTensor

            prior_covar = InterpolatedLazyTensor(
                prior_output.lazy_covariance_matrix, interp_indices, interp_values, interp_indices, interp_values
            )
            diagonal_correction = DiagLazyTensor((self.covar_diag(inputs) - prior_covar.diag()) * 0)
            test_covar = AddedDiagLazyTensor(test_covar, diagonal_correction)

        output = MultivariateNormal(test_mean, test_covar)
        return output

    def __getattr__(self, name):
        res = super(GridInducingVariationalGP, self).__getattr__(name)
        if name in self._kernels and not self._grid_mode:
            return res.base_kernel
        return res

    def __setattr__(self, name, value):
        if isinstance(value, Kernel):
            self._kernels.add(name)
            value = GridKernel(value, self.inducing_points, self.grid)
        return super(GridInducingVariationalGP, self).__setattr__(name, value)
