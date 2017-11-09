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

    def __call__(self, inputs, **kwargs):
        if self.exact_inference:
            raise RuntimeError('At the moment, the InducingPointModule only works for variational inference')

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
                self.variational_mean.data.copy_(mean_init)
                self.chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)
            # Add variational strategy
            output._variational_strategy = InducingPointStrategy(self.variational_mean,
                                                                 self.chol_variational_covar, output)

        # Posterior mode
        elif self.posterior:
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
                alpha = gpytorch.inv_matmul(induc_induc_covar, self.variational_mean - induc_mean)
                self.alpha.copy_(alpha.data)
                self.has_computed_alpha.fill_(1)
            else:
                alpha = Variable(self.alpha)

            test_mean = torch.add(test_mean, test_induc_covar.matmul(alpha))

            # Test covariance
            if isinstance(induc_test_covar, LazyVariable):
                induc_test_covar = induc_test_covar.evaluate()
            inv_product = gpytorch.inv_matmul(induc_induc_covar, induc_test_covar)
            factor = self.chol_variational_covar.matmul(inv_product)
            right_factor = factor - inv_product
            left_factor = (factor - induc_test_covar).transpose(-1, -2)

            if not isinstance(test_test_covar, LazyVariable):
                test_test_covar = NonLazyVariable(test_test_covar)
            test_covar = test_test_covar + MatmulLazyVariable(left_factor, right_factor)

            output = GaussianRandomVariable(test_mean, test_covar)

        # Prior mode
        else:
            output = super(InducingPointModule, self).__call__(inputs)

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
