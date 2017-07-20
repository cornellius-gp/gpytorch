import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.utils.interpolation import Interpolation

class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module, grid_size):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module

        self.grid_size = grid_size
        self.grid = Variable(torch.linspace(0, 1, grid_size))

    def forward(self, x1, x2, **kwargs):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            raise RuntimeError(' '.join([
                'The grid interpolation kernel can only be applied to inputs of a single dimension at this time \
                until Kronecker structure is implemented.'
            ]))

        x1 = (x1 - x1.min(0)[0].expand_as(x1)) / (x1.max(0)[0] - x1.min(0)[0]).expand_as(x1)
        x2 = (x2 - x2.min(0)[0].expand_as(x2)) / (x2.max(0)[0] - x2.min(0)[0]).expand_as(x2)

        # Explicitly compute full, dense interpolated matrix at the moment just for testing.
        W1 = Variable(Interpolation().interpolate(self.grid.data,x1.data.squeeze()).to_dense()) # Will never need gradients
        W2 = Variable(Interpolation().interpolate(self.grid.data,x2.data.squeeze()).to_dense()) # same here.

        K_UU = self.base_kernel_module(self.grid, **kwargs)

        K_XX = W1.mm(K_UU).mm(W2.t())

        return K_XX


