from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from ..distributions import MultivariateNormal
from ..models.grid_inducing_variational_gp import GridInducingVariationalGP
from ..priors import SmoothedBoxPrior


class AdditiveGridInducingVariationalGP(GridInducingVariationalGP):
    def __init__(self, grid_size, grid_bounds, num_dim, mixing_params=False, sum_output=True):
        super(AdditiveGridInducingVariationalGP, self).__init__(grid_size, grid_bounds)
        self.num_dim = num_dim
        self.sum_output = sum_output

        # Resize variational parameters to have one size per component
        variational_mean = self.variational_mean
        chol_variational_covar = self.chol_variational_covar
        variational_mean.data.resize_(*([num_dim] + list(variational_mean.size())))
        chol_variational_covar.data.resize_(*([num_dim] + list(chol_variational_covar.size())))

        # Mixing parameters
        if mixing_params:
            self.register_parameter(
                name="mixing_params",
                parameter=torch.nn.Parameter(torch.ones(num_dim) / num_dim),
                prior=SmoothedBoxPrior(-2, 2, sigma=0.01, size=num_dim),
            )

    def _compute_grid(self, inputs):
        num_data, num_dim = inputs.size()
        inputs = inputs.transpose(0, 1).contiguous().view(-1, 1)
        interp_indices, interp_values = super(AdditiveGridInducingVariationalGP, self)._compute_grid(inputs)
        interp_indices = interp_indices.view(num_dim, num_data, -1)
        interp_values = interp_values.view(num_dim, num_data, -1)

        if hasattr(self, "mixing_params"):
            interp_values = interp_values.mul(self.mixing_params.unsqueeze(1).unsqueeze(2))
        return interp_indices, interp_values

    def _initalize_variational_parameters(self, prior_output):
        batch_size = self.chol_variational_covar.size(0)
        mean_init = prior_output.mean.detach()
        mean_init += torch.randn_like(mean_init).mul_(1e-1)

        chol_covar_init = torch.eye(mean_init.size(-1), dtype=mean_init.dtype, device=mean_init.device)
        chol_covar_init = chol_covar_init.unsqueeze_(0).repeat(batch_size, 1, 1)
        chol_covar_init = chol_covar_init + torch.randn_like(chol_covar_init).mul_(1e-1)

        self.variational_mean.data.copy_(mean_init)
        self.chol_variational_covar.data.copy_(chol_covar_init)

    def covar_diag(self, inputs):
        num_data, num_dim = inputs.size()
        inputs = inputs.transpose(0, 1).contiguous().unsqueeze(-1)
        return super(AdditiveGridInducingVariationalGP, self).covar_diag(inputs)

    def prior_output(self):
        out = super(AdditiveGridInducingVariationalGP, self).prior_output()
        mean = out.mean
        covar = out.lazy_covariance_matrix.repeat(self.num_dim, 1, 1)
        return MultivariateNormal(mean, covar)

    def __call__(self, inputs, **kwargs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(-1)
        elif inputs.ndimension() != 2:
            raise RuntimeError("AdditiveGridInducingVariationalGP expects a 2d tensor.")

        num_data, num_dim = inputs.size()
        if num_dim != self.num_dim:
            raise RuntimeError("The number of dims should match the number specified.")

        output = super(AdditiveGridInducingVariationalGP, self).__call__(inputs, **kwargs)
        if self.sum_output:
            mean = output.mean.sum(0)
            covar = output.lazy_covariance_matrix.sum_batch()
            return MultivariateNormal(mean, covar)
        else:
            return output
