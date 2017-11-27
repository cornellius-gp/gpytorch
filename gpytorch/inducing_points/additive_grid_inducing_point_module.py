import gpytorch
import torch
from torch import nn
from torch.autograd import Variable
from .grid_inducing_point_module import GridInducingPointModule
from ..lazy import NonLazyVariable, SumInterpolatedLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import GridInducingPointStrategy
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
            base_lv = induc_output.covar()
            covar = SumInterpolatedLazyVariable(base_lv, interp_indices, interp_values, interp_indices, interp_values)

            return GaussianRandomVariable(mean, covar)

        else:
            variational_mean = self.variational_mean
            chol_variational_covar = self.chol_variational_covar
            induc_output = gpytorch.Module.__call__(self, Variable(self._inducing_points))
            interp_indices, interp_values = self._compute_grid(inputs)

            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized[0]:
                mean_init = induc_output.mean().data.unsqueeze(0)
                mean_init_size = list(mean_init.size())
                mean_init_size[0] = self.n_components
                mean_init = mean_init.expand(*mean_init_size)
                chol_covar_init = torch.eye(mean_init.size(-1)).type_as(mean_init).unsqueeze(0)
                chol_covar_init_size = list(chol_covar_init.size())
                chol_covar_init_size[0] = self.n_components
                chol_covar_init = chol_covar_init.expand(*chol_covar_init_size)

                variational_mean.data.copy_(mean_init)
                chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)

            # Calculate alpha vector
            if self.training:
                alpha = induc_output.mean().unsqueeze(0).expand_as(self.alpha)
            else:
                if not self.has_computed_alpha[0]:
                    alpha = variational_mean.sub(induc_output.mean())
                    self.alpha.copy_(alpha.data)
                    self.has_computed_alpha.fill_(1)
                else:
                    alpha = Variable(self.alpha)

            # Compute test mean
            # Left multiply samples by interpolation matrix
            interp_indices = Variable(interp_indices)
            interp_values = Variable(interp_values)
            if hasattr(self, 'mixing_params'):
                interp_values = interp_values.mul(self.mixing_params.unsqueeze(1).unsqueeze(2))
            test_mean = left_interp(interp_indices, interp_values, alpha.unsqueeze(-1)).sum(0).squeeze(-1)

            # Compute test covar
            if self.training:
                base_lv = induc_output.covar()
            else:
                base_lv = NonLazyVariable(self.variational_covar)
            test_covar = SumInterpolatedLazyVariable(base_lv, interp_indices, interp_values,
                                                     interp_indices, interp_values)

            output = GaussianRandomVariable(test_mean, test_covar)

            # Add variational strategy
            if self.training:
                output._variational_strategy = GridInducingPointStrategy(variational_mean,
                                                                         chol_variational_covar,
                                                                         induc_output)

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
