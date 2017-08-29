import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.utils.interpolation import Interpolation
from gpytorch.lazy import ToeplitzLazyVariable, KroneckerProductLazyVariable


class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module
        self.grid = None

    def initialize_interpolation_grid(self, grid_size, grid_bounds):
        super(GridInterpolationKernel, self).initialize_interpolation_grid(grid_size, grid_bounds)
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        self.grid = torch.zeros(len(grid_bounds), grid_size)
        for i in range(len(grid_bounds)):
            grid_diff = (grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            self.grid[i] = torch.linspace(grid_bounds[i][0] - grid_diff,
                                          grid_bounds[i][1] + grid_diff,
                                          grid_size)
        self.grid = Variable(self.grid)
        return self

    def forward(self, x1, x2, **kwargs):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            grid_size = self.grid_size
            k_UUs = Variable(torch.zeros(d, grid_size))
            Js1 = torch.zeros(d, len(x1.data), 4).long()
            Cs1 = torch.zeros(d, len(x1.data), 4)
            Js2 = torch.zeros(d, len(x1.data), 4).long()
            Cs2 = torch.zeros(d, len(x1.data), 4)
            x_max = torch.max(x1.max(0)[0], x2.max(0)[0])
            x_min = torch.max(x1.min(0)[0], x2.min(0)[0])
            grid_diff = (x_max - x_min) / (grid_size - 2)
            for i in range(d):

                scaled_grid = Variable(torch.linspace(x_min.data[i] - grid_diff.data[0],
                                                      x_max.data[i] + grid_diff.data[0],
                                                      grid_size))
                Js1[i], Cs1[i] = Interpolation().interpolate(scaled_grid.data, x1.data[:, i])
                Js2[i], Cs2[i] = Interpolation().interpolate(scaled_grid.data, x2.data[:, i])
                k_UUs[i] = self.base_kernel_module(scaled_grid[0], scaled_grid, **kwargs).squeeze()
            K_XX = KroneckerProductLazyVariable(k_UUs, Js1, Cs1, Js2, Cs2)
            return K_XX

        if self.grid is None:
            raise RuntimeError(' '.join([
                'This GridInterpolationKernel has no grid. Call initialize_interpolation_grid \
                 on a GPModel first.'
            ]))

        both_min = torch.min(x1.min(0)[0].data, x2.min(0)[0].data)[0]
        both_max = torch.max(x1.max(0)[0].data, x2.max(0)[0].data)[0]

        if both_min < self.grid_bounds[0][0] or both_max > self.grid_bounds[0][1]:
            # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
            if torch.abs(both_min - self.grid[0][0].data)[0] > 1e-7 or torch.abs(both_max - self.grid[0][-1].data)[0] > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0][0],
                                                                                              self.grid_bounds[0][1],
                                                                                              both_min,
                                                                                              both_max))

        J1, C1 = Interpolation().interpolate(self.grid.data[0], x1.data.squeeze())
        J2, C2 = Interpolation().interpolate(self.grid.data[0], x2.data.squeeze())

        k_UU = self.base_kernel_module(self.grid[0][0], self.grid[0], **kwargs).squeeze()

        K_XX = ToeplitzLazyVariable(k_UU, J1, C1, J2, C2)

        return K_XX
