import torch
from torch.autograd import Variable
from .inducing_point_module import InducingPointModule
from ..lazy import InterpolatedLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import MVNVariationalStrategy
from ..kernels import Kernel, GridInterpolationKernel
from ..utils.interpolation import Interpolation
from ..utils import left_interp


class GridInducingPointModule(InducingPointModule):
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

        super(GridInducingPointModule, self).__init__(inducing_points)
        self.grid_size = grid_size
        self.register_buffer('grid', grid)

    def __setattr__(self, name, value):
        if isinstance(value, Kernel):
            value = GridInterpolationKernel(value, self.grid_size, self.grid, self._inducing_points)
        return super(GridInducingPointModule, self).__setattr__(name, value)

    def _compute_grid(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)
        elif not inputs.ndimension() == 2:
            raise RuntimeError('Inputs must be 1 or 2 dimensional')

        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs.data)
        interp_indices = Variable(interp_indices)
        interp_values = Variable(interp_values)
        return interp_indices, interp_values

    def __call__(self, inputs, **kwargs):
        if self.exact_inference:
            if self.conditioning:
                interp_indices, interp_values = self._compute_grid(inputs)
                self.train_interp_indices = interp_indices
                self.train_interp_values = interp_values
            else:
                train_data = self.train_inputs[0].data if hasattr(self, 'train_inputs') else None
                if train_data is not None and torch.equal(inputs.data, train_data):
                    interp_indices = self.train_interp_indices
                    interp_values = self.train_interp_values
                else:
                    interp_indices, interp_values, = self._compute_grid(inputs)

            induc_output = self.prior_output()
            if not isinstance(induc_output, GaussianRandomVariable):
                raise RuntimeError('Output should be a GaussianRandomVariable')

            # Compute test mean
            # Left multiply samples by interpolation matrix
            mean = left_interp(interp_indices, interp_values, induc_output.mean())

            # Compute test covar
            base_lv = induc_output.covar()
            covar = InterpolatedLazyVariable(base_lv, interp_indices, interp_values, interp_indices, interp_values)

            return GaussianRandomVariable(mean, covar)

        else:
            interp_indices, interp_values = self._compute_grid(inputs)

            if not self.posterior:
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

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
