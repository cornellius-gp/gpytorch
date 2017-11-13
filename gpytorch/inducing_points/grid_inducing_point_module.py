import gpytorch
import torch
from copy import deepcopy
from torch.autograd import Variable
from .inducing_point_module import InducingPointModule
from ..lazy import MatmulLazyVariable, KroneckerProductLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import GridInducingPointStrategy
from ..kernels import Kernel, GridInterpolationKernel


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
            value = GridInterpolationKernel(value, self.grid_size, self.grid_bounds, self.grid)
        return super(GridInducingPointModule, self).__setattr__(name, value)

    def __call__(self, inputs, **kwargs):
        if self.exact_inference:
            output = gpytorch.Module.__call__(self, inputs)
            if not isinstance(output, GaussianRandomVariable):
                raise RuntimeError('Output should be a GaussianRandomVariable')
            return output

        # Training mode: optimizing
        if self.training:
            induc_output = gpytorch.Module.__call__(self, Variable(self._inducing_points))
            output = gpytorch.Module.__call__(self, inputs)

            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized[0]:
                mean_init = induc_output.mean().data
                chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
                self.variational_mean.data.copy_(mean_init)
                self.chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)

            # Add variational strategy
            output._variational_strategy = GridInducingPointStrategy(self.variational_mean,
                                                                     self.chol_variational_covar,
                                                                     induc_output)

        # Posterior mode
        elif self.posterior:
            output = gpytorch.Module.__call__(self, inputs)
            test_test_covar = output.covar()

            # Calculate posterior components
            if not self.has_computed_alpha[0]:
                induc_output = gpytorch.Module.__call__(self, Variable(self._inducing_points))
                alpha = self.variational_mean.sub(induc_output.mean())
                self.alpha.copy_(alpha.data)
                self.has_computed_alpha.fill_(1)
            else:
                alpha = Variable(self.alpha)

            # Hacky code for now for KroneckerProductLazyVariable. Let's change it soon.
            if isinstance(output.covar(), KroneckerProductLazyVariable):
                interp_matrix = output.covar().representation()[1]
                test_mean = gpytorch.dsmm(interp_matrix, alpha.unsqueeze(-1)).squeeze(-1)
                test_chol_covar = gpytorch.dsmm(interp_matrix, self.chol_variational_covar.t())
                test_covar = MatmulLazyVariable(test_chol_covar, test_chol_covar.t())

            else:
                # Compute test mean
                # Left multiply samples by interpolation matrix
                interp_indices = Variable(test_test_covar.J_left)
                interp_values = Variable(test_test_covar.C_left)
                mean_output = alpha.index_select(0, interp_indices.view(-1)).view(*interp_values.size())
                mean_output = mean_output.mul(interp_values)
                test_mean = mean_output.sum(-1)

                # Compute test covar
                interp_size = list(interp_indices.size()) + [self.chol_variational_covar.size(-1)]
                chol_covar_size = deepcopy(interp_size)
                chol_covar_size[-3] = self.chol_variational_covar.size()[-2]
                interp_indices_expanded = interp_indices.unsqueeze(-1).expand(*interp_size)
                test_chol_covar_output = self.chol_variational_covar.t().unsqueeze(-2)
                test_chol_covar_output = test_chol_covar_output.expand(*chol_covar_size)
                test_chol_covar_output = test_chol_covar_output.gather(-3, interp_indices_expanded)
                test_chol_covar_output = test_chol_covar_output.mul(interp_values.unsqueeze(-1).expand(interp_size))
                test_chol_covar = test_chol_covar_output.sum(-2)
                test_covar = MatmulLazyVariable(test_chol_covar, test_chol_covar.t())

            output = GaussianRandomVariable(test_mean, test_covar)
        # Prior mode
        else:
            output = gpytorch.Module.__call__(self, inputs)

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
