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
            d = x1.size(1)
            grid_var = Variable(self.grid)
            if d > 1:
                k_UUs = Variable(x1.data.new(d, self.grid_size).zero_())
                for i in range(d):
                    k_UUs[i] = self.base_kernel_module(grid_var[i, 0], grid_var[i], **kwargs).squeeze()
                K_XX = KroneckerProductLazyVariable(k_UUs)

            else:
                if gpytorch.functions.use_toeplitz:
                    k_UU = self.base_kernel_module(grid_var[0, 0], grid_var[0], **kwargs).squeeze()
                    K_XX = ToeplitzLazyVariable(k_UU)
                else:
                    for i in range(100):
                        k_UU = self.base_kernel_module(grid_var[0], grid_var[0], **kwargs).squeeze()
                    K_XX = NonLazyVariable(k_UU)

            if not self.training:
                self._cached_kernel_mat = K_XX
            return K_XX
