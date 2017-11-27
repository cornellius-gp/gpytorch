import gpytorch
import torch
from torch import nn
from torch.autograd import Variable
from ..lazy import LazyVariable, MatmulLazyVariable, NonLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import InducingPointStrategy


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

    def forward(self, inputs):
        raise NotImplementedError

    @property
    def posterior(self):
        return self.variational_params_initialized[0] and not self.training

    def train(self, mode=True):
        self.has_computed_alpha.fill_(0)
        if hasattr(self, '_variational_covar'):
            del self._variational_covar
        return super(InducingPointModule, self).train(mode)

    @property
    def variational_covar(self):
        if self.training:
            return self.chol_variational_covar.matmul(self.chol_variational_covar.transpose(-1, -2))
        else:
            if not hasattr(self, '_variational_covar'):
                transpose = self.chol_variational_covar.transpose(-1, -2)
                self._variational_covar = self.chol_variational_covar.matmul(transpose)
            return self._variational_covar

    def __call__(self, inputs, **kwargs):
        if self.exact_inference:
            raise RuntimeError('At the moment, the InducingPointModule only works for variational inference')

        variational_mean = self.variational_mean
        chol_variational_covar = self.chol_variational_covar

        # Training mode: optimizing
        if self.training:
            if not torch.equal(inputs.data, self._inducing_points):
                raise RuntimeError('At the moment, we assume that the inducing_points are the'
                                   ' training inputs.')

            output = super(InducingPointModule, self).__call__(Variable(self._inducing_points))
            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized[0]:
                mean_init = output.mean().data
                chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
                variational_mean.data.copy_(mean_init)
                chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)
            # Add variational strategy
            output._variational_strategy = InducingPointStrategy(variational_mean,
                                                                 chol_variational_covar, output)

        # Posterior mode
        else:
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
                alpha = gpytorch.inv_matmul(induc_induc_covar, variational_mean - induc_mean)
                self.alpha.copy_(alpha.data)
                self.has_computed_alpha.fill_(1)
            else:
                alpha = Variable(self.alpha)

            test_mean = torch.add(test_mean, test_induc_covar.matmul(alpha))

            # Test covariance
            if isinstance(induc_test_covar, LazyVariable):
                induc_test_covar = induc_test_covar.evaluate()
            inv_product = gpytorch.inv_matmul(induc_induc_covar, induc_test_covar)
            factor = chol_variational_covar.matmul(inv_product)
            right_factor = factor - inv_product
            left_factor = (factor - induc_test_covar).transpose(-1, -2)

            if not isinstance(test_test_covar, LazyVariable):
                test_test_covar = NonLazyVariable(test_test_covar)
            test_covar = test_test_covar + MatmulLazyVariable(left_factor, right_factor)

            output = GaussianRandomVariable(test_mean, test_covar)

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
