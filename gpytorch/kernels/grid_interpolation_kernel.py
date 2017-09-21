import math
import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.utils.interpolation import Interpolation
from gpytorch.lazy import ToeplitzLazyVariable, KroneckerProductLazyVariable


class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module

    def forward(self, x1, x2, **kwargs):
        if not self.has_grid:
            raise RuntimeError('GridInterpolationKernel requires setting the interpolation grid')

        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            grid_size = self.grid_size
            k_UUs = Variable(torch.zeros(d, grid_size))
            Js1 = torch.zeros(d, len(x1.data), 4).long()
            Cs1 = torch.zeros(d, len(x1.data), 4)
            Js2 = torch.zeros(d, len(x1.data), 4).long()
            Cs2 = torch.zeros(d, len(x1.data), 4)
            for i in range(d):
                both_min = torch.min(x1.min(0)[0].data, x2.min(0)[0].data)[i]
                both_max = torch.max(x1.max(0)[0].data, x2.max(0)[0].data)[i]
                if both_min < self.grid_bounds[i][0] or both_max > self.grid_bounds[i][1]:
                    # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
                    if math.fabs(both_min - self.grid.data[i][0]) > 1e-7:
                        raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                            Grid bounds were ({}, {}), but min = {}, \
                                            max = {}'.format(self.grid_bounds[i][0],
                                                             self.grid_bounds[i][1],
                                                             both_min,
                                                             both_max))
                    elif math.fabs(both_max - self.grid.data[i][-1]) > 1e-7:
                        raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                            Grid bounds were ({}, {}), but min = {}, \
                                            max = {}'.format(self.grid_bounds[i][0],
                                                             self.grid_bounds[i][1],
                                                             both_min,
                                                             both_max))
                Js1[i], Cs1[i] = Interpolation().interpolate(self.grid.data[i], x1.data[:, i])
                Js2[i], Cs2[i] = Interpolation().interpolate(self.grid.data[i], x2.data[:, i])
                k_UUs[i] = self.base_kernel_module(self.grid[i][0], self.grid[i], **kwargs).squeeze()
            K_XX = KroneckerProductLazyVariable(k_UUs, Js1, Cs1, Js2, Cs2)
            return K_XX

        both_min = torch.min(x1.min(0)[0].data, x2.min(0)[0].data)[0]
        both_max = torch.max(x1.max(0)[0].data, x2.max(0)[0].data)[0]

        if both_min < self.grid_bounds[0][0] or both_max > self.grid_bounds[0][1]:
            # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
            if math.fabs(both_min - self.grid.data[0][0]) > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0][0],
                                                                                              self.grid_bounds[0][1],
                                                                                              both_min,
                                                                                              both_max))
            elif math.fabs(both_max - self.grid.data[0][-1]) > 1e-7:
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

    @property
    def needs_grid(self):
        return True
