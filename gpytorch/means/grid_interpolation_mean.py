import torch
from torch.autograd import Variable
from .mean import Mean


class GridInterpolationMean(Mean):
    def __init__(self, base_mean_module, grid_size):
        super(GridInterpolationMean, self).__init__()
        self.base_mean_module = base_mean_module

        grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)

        grid_diff = grid[1] - grid[0]

        self.grid_size = grid_size + 2
        self.grid = Variable(torch.linspace(0 - grid_diff, 1 + grid_diff, grid_size + 2))

    def forward(self, input, **kwargs):
        if self.training:
            return self.base_mean_module(self.grid, **kwargs)
        else:
            return self.base_mean_module(input, **kwargs)
