import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.utils.interpolation import Interpolation
from gpytorch.lazy import ToeplitzLazyVariable

import pdb

class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module
        self.grid = None

    def initialize_interpolation_grid(self, grid_size, grid_bounds):
        super(GridInterpolationKernel, self).initialize_interpolation_grid(grid_size, grid_bounds)
        grid_size = grid_size
        grid = torch.linspace(grid_bounds[0], grid_bounds[1], grid_size - 2)

        grid_diff = grid[1] - grid[0]

        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        self.grid = Variable(torch.linspace(grid_bounds[0] - grid_diff,
                                            grid_bounds[1] + grid_diff,
                                            grid_size))

        return self

    def forward(self, x1, x2, **kwargs):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            raise RuntimeError(' '.join([
                'The grid interpolation kernel can only be applied to inputs of a single dimension at this time \
                until Kronecker structure is implemented.'
            ]))

        if self.grid is None:
            raise RuntimeError(' '.join([
                'This GridInterpolationKernel has no grid. Call initialize_interpolation_grid \
                 on a GPModel first.'
            ]))

        both_min = torch.min(x1.min(0)[0].data, x2.min(0)[0].data)[0]
        both_max = torch.max(x1.max(0)[0].data, x2.max(0)[0].data)[0]

        if both_min < self.grid_bounds[0] or both_max > self.grid_bounds[1]:
            # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
            if torch.abs(both_min - self.grid[0].data)[0] > 1e-7 or torch.abs(both_max - self.grid[-1].data)[0] > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0],
                                                                                              self.grid_bounds[1],
                                                                                              both_min,
                                                                                              both_max))

        J1, C1 = Interpolation().interpolate(self.grid.data, x1.data.squeeze())
        J2, C2 = Interpolation().interpolate(self.grid.data, x2.data.squeeze())

        k_UU = self.base_kernel_module(self.grid[0], self.grid, **kwargs).squeeze()

        K_XX = ToeplitzLazyVariable(k_UU, J1, C1, J2, C2)

        return K_XX
