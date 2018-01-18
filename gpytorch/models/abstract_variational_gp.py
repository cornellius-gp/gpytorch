import torch
from torch import nn
from torch.autograd import Variable
from ..module import Module
from ..random_variables import GaussianRandomVariable
from ..lazy import CholLazyVariable


class AbstractVariationalGP(Module):
    def __init__(self, inducing_points):
        super(AbstractVariationalGP, self).__init__()
        if not torch.is_tensor(inducing_points):
            raise RuntimeError('inducing_points must be a Tensor')
        n_inducing = inducing_points.size(0)
        self.register_buffer('inducing_points', inducing_points)
        self.register_buffer('variational_params_initialized', torch.zeros(1))
        self.register_parameter('variational_mean', nn.Parameter(torch.zeros(n_inducing)), bounds=(-1e4, 1e4))
        self.register_parameter('chol_variational_covar',
                                nn.Parameter(torch.eye(n_inducing, n_inducing)), bounds=(-100, 100))
        self.register_variational_strategy('inducing_point_strategy')

    def marginal_log_likelihood(self, likelihood, output, target, n_data=None):
        """
        Returns the marginal log likelihood of the data

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - output: (GaussianRandomVariable) - the output of the GP model
        - target: (Variable) - target
        - n_data: (int) - total number of data points in the set (required only for SGD)
        """
        if n_data is None:
            n_data = len(target)
        n_batch = target.size(0)

        log_likelihood = likelihood.log_probability(output, target).div(n_batch)
        kl_divergence = sum(variational_strategy.kl_divergence()
                            for variational_strategy in self.variational_strategies()).div(n_data)

        res = log_likelihood - kl_divergence
        return res

    def prior_output(self):
        res = super(AbstractVariationalGP, self).__call__(Variable(self.inducing_points))
        if not isinstance(res, GaussianRandomVariable):
            raise RuntimeError('%s.forward must return a GaussianRandomVariable' % self.__class__.__name__)
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
            mask = chol_variational_covar.data.new(*chol_variational_covar_size).fill_(1).triu()
            mask = Variable(mask.unsqueeze(0).expand(*([chol_variational_covar.size(0)] + chol_variational_covar_size)))

            batch_index = chol_variational_covar.data.new(batch_size).long()
            torch.arange(0, batch_size, out=batch_index)
            batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
            diag_index = chol_variational_covar.data.new(diag_size).long()
            torch.arange(0, diag_size, out=diag_index)
            diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
            diag = chol_variational_covar[batch_index, diag_index, diag_index].view(batch_size, diag_size)

            chol_variational_covar = chol_variational_covar.mul(mask)
            inside = diag.sign().unsqueeze(-1).expand_as(chol_variational_covar).mul(mask)
        else:
            raise RuntimeError('Invalid number of variational covar dimensions')

        chol_variational_covar = inside.mul(chol_variational_covar)
        return GaussianRandomVariable(self.variational_mean, CholLazyVariable(chol_variational_covar))
