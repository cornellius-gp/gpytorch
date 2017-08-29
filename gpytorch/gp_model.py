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
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        grid = torch.zeros(len(grid_bounds), grid_size)
        for i in range(len(grid_bounds)):
            grid_diff = (grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[i] = torch.linspace(grid_bounds[i][0] - grid_diff,
                                     grid_bounds[i][1] + grid_diff,
                                     grid_size)
        self.inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        for i in range(self.inducing_points.size()[0]):
            for j in range(len(grid_bounds)):
                self.inducing_points[i][j] = grid[j][int(i / pow(grid_size, j + 1))]
        self.inducing_points = Variable(self.inducing_points)
        return self

    def __call__(self, *args, **kwargs):
        output = super(GPModel, self).__call__(*args, **kwargs)
        if isinstance(output, Variable) or isinstance(output, RandomVariable) or isinstance(output, LazyVariable):
            output = (output,)
        return self.likelihood(*output)
