import gpytorch
import torch
from torch import nn
from torch.autograd import Variable
from ..lazy import CholLazyVariable, LazyVariable, MatmulLazyVariable, NonLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import MVNVariationalStrategy


class InducingPointModule(gpytorch.Module):
    def __init__(self, inducing_points):
        super(InducingPointModule, self).__init__()
        self.register_buffer('_inducing_points', inducing_points)
        n_inducing = self._inducing_points.size(0)

        self.register_parameter('variational_mean', nn.Parameter(torch.zeros(n_inducing)), bounds=(-1e4, 1e4))
        self.register_parameter('chol_variational_covar',
                                nn.Parameter(torch.eye(n_inducing, n_inducing)), bounds=(-100, 100))

        self.register_buffer('alpha', torch.Tensor(n_inducing))
        self.register_buffer('has_computed_alpha', torch.ByteTensor([0]))
        self.register_buffer('variational_params_initialized', torch.ByteTensor([0]))

        self.register_variational_strategy('inducing_point_strategy')

    def forward(self, inputs):
        raise NotImplementedError

    def prior_output(self):
        return super(InducingPointModule, self).__call__(Variable(self._inducing_points))

    @property
    def posterior(self):
        return self.variational_params_initialized[0] and not self.training

    def train(self, mode=True):
        if hasattr(self, '_variational_covar'):
            del self._variational_covar
        return super(InducingPointModule, self).train(mode)

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

    def __call__(self, inputs, **kwargs):
        if self.exact_inference:
            raise RuntimeError('At the moment, the InducingPointModule only works for variational inference')

        # Training mode: optimizing
        if self.training:
            if not torch.equal(inputs.data, self._inducing_points):
                raise RuntimeError('At the moment, we assume that the inducing_points are the'
                                   ' training inputs.')

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
            return variational_output

        # Posterior mode
        elif self.posterior:
            variational_output = self.variational_output()

            n_induc = len(self._inducing_points)
            full_inputs = torch.cat([Variable(self._inducing_points), inputs])
            full_output = super(InducingPointModule, self).__call__(full_inputs)
            full_mean, full_covar = full_output.representation()

            induc_mean = full_mean[:n_induc]
            test_mean = full_mean[n_induc:]
            induc_induc_covar = full_covar[:n_induc, :n_induc]
            induc_test_covar = full_covar[:n_induc, n_induc:]
            test_induc_covar = full_covar[n_induc:, :n_induc]
            test_test_covar = full_covar[n_induc:, n_induc:]

            # Calculate posterior components
            if not self.has_computed_alpha[0]:
                alpha = gpytorch.inv_matmul(induc_induc_covar, variational_output.mean() - induc_mean)
                self.alpha.copy_(alpha.data)
                self.has_computed_alpha.fill_(1)
            else:
                alpha = Variable(self.alpha)

            test_mean = torch.add(test_mean, test_induc_covar.matmul(alpha))

            # Test covariance
            if isinstance(induc_test_covar, LazyVariable):
                induc_test_covar = induc_test_covar.evaluate()
            inv_product = gpytorch.inv_matmul(induc_induc_covar, induc_test_covar)
            factor = variational_output.covar().chol_matmul(inv_product)
            right_factor = factor - inv_product
            left_factor = (factor - induc_test_covar).transpose(-1, -2)

            if not isinstance(test_test_covar, LazyVariable):
                test_test_covar = NonLazyVariable(test_test_covar)
            test_covar = test_test_covar + MatmulLazyVariable(left_factor, right_factor)

            output = GaussianRandomVariable(test_mean, test_covar)

            return output

        # Prior mode
        else:
            return super(InducingPointModule, self).__call__(inputs)
