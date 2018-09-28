from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from ..distributions import MultivariateNormal
from ..lazy import CholLazyTensor
from ..module import Module


class AbstractVariationalGP(Module):
    def __init__(self, inducing_points):
        super(AbstractVariationalGP, self).__init__()
        if not torch.is_tensor(inducing_points):
            raise RuntimeError("inducing_points must be a Tensor")
        n_inducing = inducing_points.size(0)
        self.register_buffer("inducing_points", inducing_points)
        self.register_buffer("variational_params_initialized", torch.tensor(0))
        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(torch.zeros(n_inducing)))
        self.register_parameter(
            name="chol_variational_covar", parameter=torch.nn.Parameter(torch.eye(n_inducing, n_inducing))
        )
        self.register_variational_strategy("inducing_point_strategy")

    def covar_diag(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)
        orig_size = list(inputs.size())

        # Resize inputs so that everything is batch
        inputs = inputs.unsqueeze(-2).view(-1, 1, inputs.size(-1))

        # Get diagonal of covar
        res = super(AbstractVariationalGP, self).__call__(inputs)
        covar_diag = res.lazy_covariance_matrix.diag()
        covar_diag = covar_diag.view(orig_size[:-1])

        return covar_diag

    def prior_output(self):
        res = super(AbstractVariationalGP, self).__call__(self.inducing_points)
        if not isinstance(res, MultivariateNormal):
            raise RuntimeError("{}.forward must return a MultivariateNormal".format(self.__class__.__name__))

        res = MultivariateNormal(res.mean, res.lazy_covariance_matrix.evaluate_kernel())
        return res

    def variational_output(self):
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
