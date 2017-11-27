import gpytorch
import torch
import math
from torch.autograd import Variable
from .inducing_point_module import InducingPointModule
from ..lazy import NonLazyVariable, KroneckerProductLazyVariable, InterpolatedLazyVariable, MatmulLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import GridInducingPointStrategy
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
        self.grid_bounds = grid_bounds
        self.register_buffer('grid', grid)

    def __setattr__(self, name, value):
        if isinstance(value, Kernel):
            value = GridInterpolationKernel(value, self.grid_size, self.grid, self._inducing_points)
        return super(GridInducingPointModule, self).__setattr__(name, value)

    def _compute_grid(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)
        d = inputs.size(1)

        if d > 1:
            interp_indices = inputs.data.new(d, len(inputs.data), 4).zero_().long()
            interp_values = inputs.data.new(d, len(inputs.data), 4).zero_().float()
            for i in range(d):
                inputs_min = inputs.min(0)[0].data[i]
                inputs_max = inputs.max(0)[0].data[i]
                if inputs_min < self.grid_bounds[i][0] or inputs_max > self.grid_bounds[i][1]:
                    # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
                    if math.fabs(inputs_min - self.grid[i, 0]) > 1e-7:
                        raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                            Grid bounds were ({}, {}), but min = {}, \
                                            max = {}'.format(self.grid_bounds[i][0],
                                                             self.grid_bounds[i][1],
                                                             inputs_min,
                                                             inputs_max))
                    elif math.fabs(inputs_max - self.grid[i, -1]) > 1e-7:
                        raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                            Grid bounds were ({}, {}), but min = {}, \
                                            max = {}'.format(self.grid_bounds[i][0],
                                                             self.grid_bounds[i][1],
                                                             inputs_min,
                                                             inputs_max))
                dim_interp_indices, dim_interp_values = Interpolation().interpolate(self.grid[i], inputs.data[:, i])
                interp_indices[i].copy_(dim_interp_indices)
                interp_values[i].copy_(dim_interp_values)
            return interp_indices, interp_values

        inputs_min = inputs.min(0)[0].data[0]
        inputs_max = inputs.max(0)[0].data[0]
        if inputs_min < self.grid_bounds[0][0] or inputs_max > self.grid_bounds[0][1]:
            # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
            if math.fabs(inputs_min - self.grid[0, 0]) > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0][0],
                                                                                              self.grid_bounds[0][1],
                                                                                              inputs_min,
                                                                                              inputs_max))
            elif math.fabs(inputs_max - self.grid[0, -1]) > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0][0],
                                                                                              self.grid_bounds[0][1],
                                                                                              inputs_min,
                                                                                              inputs_max))
        interp_indices, interp_values = Interpolation().interpolate(self.grid[0], inputs.data.squeeze())
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

            induc_output = gpytorch.Module.__call__(self, Variable(self._inducing_points))
            if not isinstance(induc_output, GaussianRandomVariable):
                raise RuntimeError('Output should be a GaussianRandomVariable')

            if isinstance(induc_output.covar(), KroneckerProductLazyVariable):
                covar = KroneckerProductLazyVariable(induc_output.covar().columns, interp_indices, interp_values,
                                                     interp_indices, interp_values)
                interp_matrix = covar.representation()[1]
                mean = gpytorch.dsmm(interp_matrix, induc_output.mean().unsqueeze(-1)).squeeze(-1)

            else:
                # Compute test mean
                # Left multiply samples by interpolation matrix
                interp_indices = Variable(interp_indices)
                interp_values = Variable(interp_values)
                mean = left_interp(interp_indices, interp_values, induc_output.mean())

                # Compute test covar
                base_lv = induc_output.covar()
                covar = InterpolatedLazyVariable(base_lv, interp_indices, interp_values, interp_indices, interp_values)

            return GaussianRandomVariable(mean, covar)

        else:
            variational_mean = self.variational_mean
            chol_variational_covar = self.chol_variational_covar
            induc_output = gpytorch.Module.__call__(self, Variable(self._inducing_points))
            interp_indices, interp_values = self._compute_grid(inputs)

            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized[0]:
                mean_init = induc_output.mean().data
                chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
                variational_mean.data.copy_(mean_init)
                chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)

            # Calculate alpha vector
            if self.training:
                alpha = induc_output.mean()
            else:
                if not self.has_computed_alpha[0]:
                    alpha = variational_mean.sub(induc_output.mean())
                    self.alpha.copy_(alpha.data)
                    self.has_computed_alpha.fill_(1)
                else:
                    alpha = Variable(self.alpha)

            if isinstance(induc_output.covar(), KroneckerProductLazyVariable):
                test_covar = KroneckerProductLazyVariable(induc_output.covar().columns, interp_indices, interp_values,
                                                          interp_indices, interp_values)
                interp_matrix = test_covar.representation()[1]
                test_mean = gpytorch.dsmm(interp_matrix, alpha.unsqueeze(-1)).squeeze(-1)
                if not self.training:
                    test_chol_covar = gpytorch.dsmm(interp_matrix, chol_variational_covar)
                    test_covar = MatmulLazyVariable(test_chol_covar, test_chol_covar.transpose(-2, -1))

            else:
                # Compute test mean
                # Left multiply samples by interpolation matrix
                interp_indices = Variable(interp_indices)
                interp_values = Variable(interp_values)
                test_mean = left_interp(interp_indices, interp_values, alpha)

                # Compute test covar
                if self.training:
                    base_lv = induc_output.covar()
                else:
                    base_lv = NonLazyVariable(self.variational_covar)
                test_covar = InterpolatedLazyVariable(base_lv, interp_indices, interp_values,
                                                      interp_indices, interp_values)

            output = GaussianRandomVariable(test_mean, test_covar)

            # Add variational strategy
            if self.training:
                output._variational_strategy = GridInducingPointStrategy(variational_mean,
                                                                         chol_variational_covar,
                                                                         induc_output)

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
