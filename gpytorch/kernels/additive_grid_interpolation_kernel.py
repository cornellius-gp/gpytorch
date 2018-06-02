from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
from .grid_interpolation_kernel import GridInterpolationKernel
from ..utils import Interpolation


class AdditiveGridInterpolationKernel(GridInterpolationKernel):
    def __init__(self, base_kernel_module, grid_size, grid_bounds, n_components, active_dims=None):
        super(AdditiveGridInterpolationKernel, self).__init__(
            base_kernel_module, grid_size, grid_bounds, active_dims=active_dims
        )
        self.n_components = n_components

    def _compute_grid(self, inputs):
        inputs = inputs.view(inputs.size(0), inputs.size(1), self.n_components, -1)
        batch_size, n_data, n_components, n_dimensions = inputs.size()
        inputs = inputs.transpose(0, 2).contiguous().view(n_components * batch_size * n_data, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(Variable(self.grid), inputs)
        interp_indices = interp_indices.view(n_components * batch_size, n_data, -1)
        interp_values = interp_values.view(n_components * batch_size, n_data, -1)
        return interp_indices, interp_values

    def _inducing_forward(self):
        res = super(AdditiveGridInterpolationKernel, self)._inducing_forward()
        return res.repeat(self.n_components, 1, 1)

    def forward(self, x1, x2):
        res = super(AdditiveGridInterpolationKernel, self).forward(x1, x2)
        return res.sum_batch(sum_batch_size=self.n_components)
