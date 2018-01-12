import torch
from torch.autograd import Variable
from ..random_variables import GaussianRandomVariable
from ..lazy import InterpolatedLazyVariable
from ..variational import MVNVariationalStrategy
from ..kernels.kernel import Kernel
from ..kernels.grid_kernel import GridKernel
from ..utils import Interpolation, left_interp
from .abstract_variational_gp import AbstractVariationalGP


class GridInducingVariationalGP(AbstractVariationalGP):
    def __init__(self, grid_size, grid_bounds):
        grid = torch.zeros(len(grid_bounds), grid_size)
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[i] = torch.linspace(grid_bounds[i][0] - grid_diff,
                                     grid_bounds[i][1] + grid_diff,
                                     grid_size)

        inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        prev_points = None
        for i in range(len(grid_bounds)):
            for j in range(grid_size):
                inducing_points[j * grid_size ** i:(j + 1) * grid_size ** i, i].fill_(grid[i, j])
                if prev_points is not None:
                    inducing_points[j * grid_size ** i:(j + 1) * grid_size ** i, :i].copy_(prev_points)
            prev_points = inducing_points[:grid_size ** (i + 1), :(i + 1)]

        super(GridInducingVariationalGP, self).__init__(inducing_points)
        self.register_buffer('grid', grid)

    def _compute_grid(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs.data)
        interp_indices = Variable(interp_indices)
        interp_values = Variable(interp_values)
        return interp_indices, interp_values

    def __call__(self, inputs, **kwargs):
        interp_indices, interp_values = self._compute_grid(inputs)

        if self.training:
            prior_output = self.prior_output()
            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized[0]:
                mean_init = prior_output.mean().data
                chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
                self.variational_mean.data.copy_(mean_init)
                self.chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)

            variational_output = self.variational_output()
            new_variational_strategy = MVNVariationalStrategy(variational_output, prior_output)
            self.update_variational_strategy('inducing_point_strategy', new_variational_strategy)
        else:
            variational_output = self.variational_output()

        # Compute test mean
        # Left multiply samples by interpolation matrix
        test_mean = left_interp(interp_indices, interp_values, variational_output.mean().unsqueeze(-1))
        test_mean = test_mean.squeeze(-1)

        # Compute test covar
        test_covar = InterpolatedLazyVariable(variational_output.covar(), interp_indices, interp_values,
                                              interp_indices, interp_values)

        output = GaussianRandomVariable(test_mean, test_covar)
        return output

    def __setattr__(self, name, value):
        if isinstance(value, Kernel):
            value = GridKernel(value, self.inducing_points, self.grid)
        return super(GridInducingVariationalGP, self).__setattr__(name, value)
