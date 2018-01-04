import gpytorch
import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.lazy import ToeplitzLazyVariable, KroneckerProductLazyVariable, NonLazyVariable


class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module, grid_size, grid, inducing_points):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module
        self.grid_size = grid_size
        self.register_buffer('grid', grid)
        if inducing_points.ndimension() == 2:
            inducing_points = inducing_points.unsqueeze(0)
        self.register_buffer('_inducing_points', inducing_points)

    def train(self, mode=True):
        if hasattr(self, '_cached_kernel_mat'):
            del self._cached_kernel_mat
        return super(GridInterpolationKernel, self).train(mode)

    def forward(self, x1, x2, **kwargs):
        if not torch.equal(x1.data, self._inducing_points) or \
                not torch.equal(x1.data, self._inducing_points):
            raise RuntimeError('The kernel should only receive the inducing points as input')

        if not self.training and hasattr(self, '_cached_kernel_mat'):
            return self._cached_kernel_mat
        else:
            d = x1.size(-1)
            grid_var = Variable(self.grid)

            if gpytorch.functions.use_toeplitz:
                first_item = grid_var[:, 0].contiguous().view(d, 1, 1)
                k_UU = self.base_kernel_module(first_item, grid_var.view(d, -1, 1), **kwargs)
                K_XXs = [ToeplitzLazyVariable(k_UU[i:i + 1].squeeze(-2)) for i in range(d)]
            else:
                k_UU = self.base_kernel_module(grid_var.view(d, -1, 1), grid_var.view(d, -1, 1), **kwargs)
                K_XXs = [NonLazyVariable(k_UU[i:i + 1]) for i in range(d)]

            if d > 1:
                K_XX = KroneckerProductLazyVariable(*K_XXs)
            else:
                K_XX = K_XXs[0]

            if not self.training:
                self._cached_kernel_mat = K_XX
            return K_XX
