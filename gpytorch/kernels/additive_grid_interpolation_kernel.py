from torch.autograd import Variable
from .grid_interpolation_kernel import GridInterpolationKernel
from ..lazy import SumBatchLazyVariable
from ..utils import Interpolation


class AdditiveGridInterpolationKernel(GridInterpolationKernel):
    def __init__(self, base_kernel_module, grid_size, grid_bounds, n_components):
        super(AdditiveGridInterpolationKernel, self).__init__(base_kernel_module, grid_size, grid_bounds)
        self.n_components = n_components

    def _compute_grid(self, inputs):
        inputs = inputs.view(inputs.size(0), inputs.size(1), self.n_components, -1)
        batch_size, n_data, n_components, n_dimensions = inputs.size()
        inputs = inputs.transpose(0, 2).contiguous().view(n_components * batch_size * n_data, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs.data)
        interp_indices = Variable(interp_indices).view(n_components * batch_size, n_data, -1)
        interp_values = Variable(interp_values).view(n_components * batch_size, n_data, -1)
        return interp_indices, interp_values

    def _inducing_forward(self):
        res = super(AdditiveGridInterpolationKernel, self)._inducing_forward()
        return res.repeat(self.n_components, 1, 1)

    def forward(self, x1, x2):
        res = super(AdditiveGridInterpolationKernel, self).forward(x1, x2)
        return SumBatchLazyVariable(res, sum_batch_size=self.n_components)
