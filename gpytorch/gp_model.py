import gpytorch
import torch
from torch.autograd import Variable
from .random_variables import RandomVariable
from .lazy import LazyVariable


class GPModel(gpytorch.Module):
    def __init__(self, likelihood):
        super(GPModel, self).__init__()
        self._parameter_groups = {}
        self.likelihood = likelihood
        self.inducing_points = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_interpolation_grid(self, grid_size, grid_bounds):
        super(GPModel, self).initialize_interpolation_grid(grid_size, grid_bounds)
        grid_size = grid_size
        grid = torch.linspace(grid_bounds[0], grid_bounds[1], grid_size - 2)

        grid_diff = grid[1] - grid[0]

        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        self.inducing_points = Variable(torch.linspace(grid_bounds[0] - grid_diff, 
                                                       grid_bounds[1] + grid_diff,
                                                       grid_size))

        return self

    def __call__(self, *args, **kwargs):
        output = super(GPModel, self).__call__(*args, **kwargs)
        if isinstance(output, Variable) or isinstance(output, RandomVariable) or isinstance(output, LazyVariable):
            output = (output,)
        return self.likelihood(*output)
