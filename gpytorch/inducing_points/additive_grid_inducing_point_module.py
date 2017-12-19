import gpytorch
import torch
from torch import nn
from torch.autograd import Variable
from .grid_inducing_point_module import GridInducingPointModule
from ..lazy import CholLazyVariable, InterpolatedLazyVariable, SumBatchLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import MVNVariationalStrategy
from ..utils import left_interp


class AdditiveGridInducingPointModule(GridInducingPointModule):
    def __init__(self, grid_size, grid_bounds, n_components, mixing_params=False):
        super(AdditiveGridInducingPointModule, self).__init__(grid_size, grid_bounds)
        self.n_components = n_components

        # Resize variational parameters to have one size per component
        self.alpha.resize_(*([n_components] + list(self.alpha.size())))
        variational_mean = self.variational_mean
        chol_variational_covar = self.chol_variational_covar
        variational_mean.data.resize_(*([n_components] + list(variational_mean.size())))
        chol_variational_covar.data.resize_(*([n_components] + list(chol_variational_covar.size())))

        # Mixing parameters
        if mixing_params:
            self.register_parameter('mixing_params',
                                    nn.Parameter(torch.Tensor(n_components).fill_(1. / n_components)),
                                    bounds=(-2, 2))

    def _compute_grid(self, inputs):
        n_data, n_components, n_dimensions = inputs.size()
        inputs = inputs.transpose(0, 1).contiguous().view(n_components * n_data, n_dimensions)
        interp_indices, interp_values = super(AdditiveGridInducingPointModule, self)._compute_grid(inputs)
        interp_indices = interp_indices.view(n_components, n_data, -1)
        interp_values = interp_values.view(n_components, n_data, -1)
        return interp_indices, interp_values

    def __call__(self, inputs, **kwargs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(-1).unsqueeze(-1)
        elif inputs.ndimension() == 2:
            inputs = inputs.unsqueeze(-1)
        elif inputs.ndimension() != 3:
            raise RuntimeError('AdditiveGridInducingPointModule expects a 3d tensor.')

        n_data, n_components, n_dimensions = inputs.size()
        if n_dimensions != len(self.grid_bounds):
            raise RuntimeError('The number of dimensions should match the inducing points number of dimensions.')
        if n_components != self.n_components:
            raise RuntimeError('The number of components should match the number specified.')
        if n_dimensions != 1:
            raise RuntimeError('At the moment, AdditiveGridInducingPointModule only supports 1d'
                               ' (Toeplitz) interpolation.')

        if self.exact_inference:
            if self.conditioning:
                interp_indices, interp_values = self._compute_grid(inputs)
                self.train_interp_indices = interp_indices
                self.train_interp_values = interp_values
            else:
                train_data = self.train_inputs[0].data if hasattr(self, 'train_inputs') else None
                if train_data is not None and torch.equal(inputs.data, train_data):
                    interp_indices = self.train_interp_indices
                    interp_values = self.train_interp_values
                else:
                    interp_indices, interp_values, = self._compute_grid(inputs)

            induc_output = gpytorch.Module.__call__(self, Variable(self._inducing_points))
            if not isinstance(induc_output, GaussianRandomVariable):
                raise RuntimeError('Output should be a GaussianRandomVariable')

            # Compute test mean
            # Left multiply samples by interpolation matrix
            interp_indices = Variable(interp_indices)
            interp_values = Variable(interp_values)
            if hasattr(self, 'mixing_params'):
                interp_values = interp_values.mul(self.mixing_params.unsqueeze(1).unsqueeze(2))
            mean = left_interp(interp_indices, interp_values, induc_output.mean()).sum(0)

            # Compute test covar
            base_lv = InterpolatedLazyVariable(induc_output.covar(), interp_indices, interp_values,
                                               interp_indices, interp_values)
            covar = SumBatchLazyVariable(base_lv)

            return GaussianRandomVariable(mean, covar)

        else:
            interp_indices, interp_values = self._compute_grid(inputs)

            if not self.posterior:
                prior_output = self.prior_output()
                # Initialize variational parameters, if necessary
                if not self.variational_params_initialized[0]:
                    mean_init = prior_output.mean().data
                    chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
                    self.variational_mean.data.copy_(mean_init)
                    self.chol_variational_covar.data.copy_(chol_covar_init)
                    self.variational_params_initialized.fill_(1)

                variational_output = self.variational_output()
                new_variational_strategy = MVNVariationalStrategy(variational_output, prior_output)
                self.update_variational_strategy('inducing_point_strategy', new_variational_strategy)
            else:
                variational_output = self.variational_output()

            # Compute test mean
            # Left multiply samples by interpolation matrix
            interp_indices = Variable(interp_indices)
            interp_values = Variable(interp_values)
            if hasattr(self, 'mixing_params'):
                interp_values = interp_values.mul(self.mixing_params.unsqueeze(1).unsqueeze(2))
            test_mean = left_interp(interp_indices, interp_values,
                                    variational_output.mean().unsqueeze(-1)).sum(0).squeeze(-1)

            # Compute test covar
            test_chol_covar = left_interp(interp_indices, interp_values, variational_output.covar().lhs)
            test_covar = SumBatchLazyVariable(CholLazyVariable(test_chol_covar))

            output = GaussianRandomVariable(test_mean, test_covar)

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
