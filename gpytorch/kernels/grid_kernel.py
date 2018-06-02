from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
from .kernel import Kernel
from ..lazy import ToeplitzLazyVariable, KroneckerProductLazyVariable
from .. import settings


class GridKernel(Kernel):
    def __init__(self, base_kernel_module, inducing_points, grid, active_dims=None):
        super(GridKernel, self).__init__(active_dims=active_dims)
        self.base_kernel_module = base_kernel_module
        if inducing_points.ndimension() != 2:
            raise RuntimeError("Inducing points should be 2 dimensional")
        self.register_buffer("inducing_points", inducing_points.unsqueeze(0))
        self.register_buffer("grid", grid)

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(GridKernel, self).train(mode)

    def forward(self, x1, x2, **kwargs):
        if not torch.equal(x1.data, self.inducing_points) or not torch.equal(x2.data, self.inducing_points):
            raise RuntimeError("The kernel should only receive the inducing points as input")

        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat

        else:
            n_dim = x1.size(-1)
            grid_var = Variable(self.grid.view(n_dim, -1, 1))

            if settings.use_toeplitz.on():
                first_item = grid_var[:, 0:1].contiguous()
                covar_columns = self.base_kernel_module(first_item, grid_var, **kwargs).evaluate()
                covars = [ToeplitzLazyVariable(covar_columns[i : i + 1].squeeze(-2)) for i in range(n_dim)]
            else:
                grid_var = grid_var.view(n_dim, -1, 1)
                covars = self.base_kernel_module(grid_var, grid_var, **kwargs).evaluate_kernel()
                covars = [covars[i : i + 1] for i in range(n_dim)]

            if n_dim > 1:
                covar = KroneckerProductLazyVariable(*covars)
            else:
                covar = covars[0]

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
