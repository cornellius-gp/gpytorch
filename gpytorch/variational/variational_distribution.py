from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from ..lazy import CholLazyTensor
from ..distributions import MultivariateNormal
from ..module import Module


class VariationalDistribution(Module):
    def __init__(self, num_inducing_points, batch_size=None):
        super(VariationalDistribution, self).__init__()
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)
        if batch_size is not None:
            mean_init = mean_init.repeat(batch_size, 1)
            covar_init = covar_init.repeat(batch_size, 1, 1)
        mean_init += torch.randn_like(mean_init).mul(1e-1)
        covar_init += torch.randn_like(covar_init).mul(1e-1)

        self.register_parameter(name="variational_mean",
                                parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar",
                                parameter=torch.nn.Parameter(covar_init))

        self.register_buffer("variational_params_initialized", torch.tensor(0))

    @property
    def variational_distribution(self):
        chol_variational_covar = self.chol_variational_covar

        # Negate each row with a negative diagonal (the Cholesky decomposition
        # of a matrix requires that the diagonal elements be positive).
        if chol_variational_covar.ndimension() == 2:
            chol_variational_covar = chol_variational_covar.triu()
            inside = chol_variational_covar.diag().sign().unsqueeze(1).expand_as(chol_variational_covar).triu()
        elif chol_variational_covar.ndimension() == 3:
            batch_size, diag_size, _ = chol_variational_covar.size()

            # Batch mode
            chol_variational_covar_size = list(chol_variational_covar.size())[-2:]
            mask = torch.ones(
                *chol_variational_covar_size, dtype=chol_variational_covar.dtype, device=chol_variational_covar.device
            ).triu_()
            mask = mask.unsqueeze(0).expand(*([chol_variational_covar.size(0)] + chol_variational_covar_size))

            batch_index = torch.arange(0, batch_size, dtype=torch.long, device=mask.device)
            batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
            diag_index = torch.arange(0, diag_size, dtype=torch.long, device=mask.device)
            diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
            diag = chol_variational_covar[batch_index, diag_index, diag_index].view(batch_size, diag_size)

            chol_variational_covar = chol_variational_covar.mul(mask)
            inside = diag.sign().unsqueeze(-1).expand_as(chol_variational_covar).mul(mask)
        else:
            raise RuntimeError("Invalid number of variational covar dimensions")

        chol_variational_covar = inside.mul(chol_variational_covar)
        variational_covar = CholLazyTensor(chol_variational_covar.transpose(-1, -2))
        return MultivariateNormal(self.variational_mean, variational_covar)

    def forward(self, *args, **kwargs):
        raise RuntimeError('VariationalDistribution is not intended to be called!')
