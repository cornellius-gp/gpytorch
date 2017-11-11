import math
import gpytorch
import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.utils.interpolation import Interpolation
from ..lazy import ToeplitzLazyVariable, KroneckerProductLazyVariable, InterpolatedLazyVariable, \
        NonLazyVariable


class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module, grid_size, grid_bounds, grid=None):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds

        if grid is None:
            grid = torch.zeros(len(grid_bounds), grid_size)
            for i in range(len(grid_bounds)):
                grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
                grid[i] = torch.linspace(grid_bounds[i][0] - grid_diff,
                                         grid_bounds[i][1] + grid_diff,
                                         grid_size)
        self.register_buffer('grid', grid)

    def _compute_grid(self, x1, x2):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            Js1 = x1.data.new(d, len(x1.data), 4).zero_().long()
            Cs1 = x1.data.new(d, len(x1.data), 4).zero_()
            Js2 = x1.data.new(d, len(x1.data), 4).zero_().long()
            Cs2 = x1.data.new(d, len(x1.data), 4).zero_()
            for i in range(d):
                both_min = torch.min(x1.min(0)[0].data, x2.min(0)[0].data)[i]
                both_max = torch.max(x1.max(0)[0].data, x2.max(0)[0].data)[i]
                if both_min < self.grid_bounds[i][0] or both_max > self.grid_bounds[i][1]:
                    # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
                    if math.fabs(both_min - self.grid[i, 0]) > 1e-7:
                        raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                            Grid bounds were ({}, {}), but min = {}, \
                                            max = {}'.format(self.grid_bounds[i][0],
                                                             self.grid_bounds[i][1],
                                                             both_min,
                                                             both_max))
                    elif math.fabs(both_max - self.grid[i, -1]) > 1e-7:
                        raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                            Grid bounds were ({}, {}), but min = {}, \
                                            max = {}'.format(self.grid_bounds[i][0],
                                                             self.grid_bounds[i][1],
                                                             both_min,
                                                             both_max))
                Js1[i], Cs1[i] = Interpolation().interpolate(self.grid[i], x1.data[:, i])
                Js2[i], Cs2[i] = Interpolation().interpolate(self.grid[i], x2.data[:, i])
            return Js1, Cs1, Js2, Cs2

        both_min = torch.min(x1.min(0)[0].data, x2.min(0)[0].data)[0]
        both_max = torch.max(x1.max(0)[0].data, x2.max(0)[0].data)[0]

        if both_min < self.grid_bounds[0][0] or both_max > self.grid_bounds[0][1]:
            # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
            if math.fabs(both_min - self.grid[0, 0]) > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0][0],
                                                                                              self.grid_bounds[0][1],
                                                                                              both_min,
                                                                                              both_max))
            elif math.fabs(both_max - self.grid[0, -1]) > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0][0],
                                                                                              self.grid_bounds[0][1],
                                                                                              both_min,
                                                                                              both_max))
        J1, C1 = Interpolation().interpolate(self.grid[0], x1.data.squeeze())
        J2, C2 = Interpolation().interpolate(self.grid[0], x2.data.squeeze())
        return J1, C1, J2, C2

    def forward(self, x1, x2, **kwargs):
        n, d = x1.size()
        grid_size = self.grid_size

        if self.conditioning:
            J1, C1, J2, C2 = self._compute_grid(x1, x2)
            self.train_J1 = J1
            self.train_C1 = C1
            self.train_J2 = J2
            self.train_C2 = C2
        else:
            train_data = self.train_inputs[0].data if hasattr(self, 'train_inputs') else None
            if train_data is not None and torch.equal(x1.data, train_data) and torch.equal(x2.data, train_data):
                J1 = self.train_J1
                C1 = self.train_C1
                J2 = self.train_J2
                C2 = self.train_C2
            else:
                J1, C1, J2, C2 = self._compute_grid(x1, x2)

        grid_var = Variable(self.grid)
        if d > 1:
            k_UUs = Variable(x1.data.new(d, grid_size).zero_())
            for i in range(d):
                k_UUs[i] = self.base_kernel_module(grid_var[i, 0], grid_var[i], **kwargs).squeeze()
            K_XX = KroneckerProductLazyVariable(k_UUs, J1, C1, J2, C2)
        else:
            if gpytorch.functions.use_toeplitz:
                k_UU = self.base_kernel_module(grid_var[0, 0], grid_var[0], **kwargs).squeeze()
                K_XX = InterpolatedLazyVariable(ToeplitzLazyVariable(k_UU), Variable(J1), Variable(C1),
                                                Variable(J2), Variable(C2))
            else:
                k_UU = self.base_kernel_module(grid_var[0], grid_var[0], **kwargs).squeeze()
                K_XX = InterpolatedLazyVariable(NonLazyVariable(k_UU), Variable(J1), Variable(C1),
                                                Variable(J2), Variable(C2))

        return K_XX
