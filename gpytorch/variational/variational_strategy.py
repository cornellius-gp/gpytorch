import torch
import gpytorch
from torch.autograd import Variable


class VariationalStrategy(object):
    def __init__(self, variational_mean, chol_variational_covar, inducing_output):
        self.variational_mean = variational_mean

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

        self.chol_variational_covar = inside.mul(chol_variational_covar)

        self.inducing_output = inducing_output

    def mvn_kl_divergence(self):
        mean_diffs = self.inducing_output.mean() - self.variational_mean
        chol_variational_covar = self.chol_variational_covar

        if chol_variational_covar.ndimension() == 2:
            matrix_diag = chol_variational_covar.diag()
        elif chol_variational_covar.ndimension() == 3:
            batch_size, diag_size, _ = chol_variational_covar.size()
            batch_index = chol_variational_covar.data.new(batch_size).long()
            torch.arange(0, batch_size, out=batch_index)
            batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
            diag_index = chol_variational_covar.data.new(diag_size).long()
            torch.arange(0, diag_size, out=diag_index)
            diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
            matrix_diag = chol_variational_covar[batch_index, diag_index, diag_index].view(batch_size, diag_size)
        else:
            raise RuntimeError('Invalid number of variational covar dimensions')

        logdet_variational_covar = matrix_diag.log().sum() * 2
        trace_logdet_quad_form = gpytorch.trace_logdet_quad_form(mean_diffs, self.chol_variational_covar,
                                                                 gpytorch.add_jitter(self.inducing_output.covar()))

        # Compute the KL Divergence.
        res = 0.5 * (trace_logdet_quad_form - logdet_variational_covar - len(mean_diffs))
        return res

    def variational_samples(self, output, n_samples=None):
        raise NotImplementedError
