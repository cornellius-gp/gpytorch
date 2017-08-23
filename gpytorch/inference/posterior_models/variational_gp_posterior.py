import torch
import gpytorch
from torch import nn
from torch.autograd import Variable
from gpytorch.random_variables import GaussianRandomVariable
from .gp_posterior import _GPPosterior


class _VariationalGPPosterior(_GPPosterior):
    def __init__(self, prior_model, inducing_points, train_xs=None, train_y=None):
        super(_VariationalGPPosterior, self).__init__(prior_model.likelihood)
        self.prior_model = prior_model

        self.inducing_points = inducing_points
        self.num_inducing = len(self.inducing_points[0])
        # Alpha cache - for computing mean
        self.register_buffer('alpha', torch.Tensor())

        if train_xs is not None and train_y is not None:
            self.update_data(train_xs, train_y)

        output = self.forward(*self.inducing_points)
        mean_init, chol_covar_init = self.initialize_variational_parameters(output)

        self.register_parameter('variational_mean',
                                nn.Parameter(mean_init),
                                bounds=(-1e4, 1e4))

        self.register_parameter('chol_variational_covar',
                                nn.Parameter(chol_covar_init.triu_()),
                                bounds=(-100, 100))

    def initialize_variational_parameters(self, var_output):
        mean_init = var_output.mean().data
        chol_covar_init = torch.eye(len(mean_init))
        return mean_init.contiguous(), chol_covar_init.contiguous()

    def update_data(self, train_xs, train_y):
        if isinstance(train_xs, Variable) or isinstance(train_xs, torch._TensorBase):
            train_xs = (train_xs, )
        train_xs = [input.data if isinstance(input, Variable) else input for input in train_xs]
        train_y = train_y.data if isinstance(train_y, Variable) else train_y

        self.train_xs = []
        for i, train_x in enumerate(train_xs):
            if hasattr(self, 'train_x_%d' % i):
                getattr(self, 'train_x_%d').resize_as_(train_x).copy_(train_x)
            else:
                self.register_buffer('train_x_%d' % i, train_x)
            self.train_xs.append(getattr(self, 'train_x_%d' % i))

        if hasattr(self, 'train_y'):
            self.train_y.resize_as_(train_y).copy_(train_y)
        else:
            self.register_buffer('train_y', train_y)

        # Reset alpha cache
        self.alpha.resize_(0)
        return self

    def forward(self, *inputs, **params):
        if not self.training:
            inducing_point_vars = [inducing_pt for inducing_pt in self.inducing_points]
            full_inputs = [torch.cat([inducing_point_var, input])
                           for inducing_point_var, input in zip(inducing_point_vars, inputs)]
        else:
            full_inputs = inputs

        gaussian_rv_output = self.prior_model.forward(*full_inputs, **params)
        full_mean, full_covar = gaussian_rv_output.representation()

        if not self.training:
            # Get mean/covar components
            n = self.num_inducing
            test_mean = full_mean[n:]
            induc_induc_covar = full_covar[:n, :n]
            induc_test_covar = full_covar[:n, n:]
            test_induc_covar = full_covar[n:, :n]
            test_test_covar = full_covar[n:, n:]

            # Calculate posterior components
            if self.alpha.numel():
                alpha = Variable(self.alpha)
            else:
                alpha = gpytorch.variational_posterior_alpha(induc_induc_covar, self.variational_mean)
                self.alpha.resize_as_(alpha.data).copy_(alpha.data)
            test_mean = gpytorch.variational_posterior_mean(test_induc_covar, alpha)
            test_covar = gpytorch.variational_posterior_covar(test_induc_covar, induc_test_covar,
                                                              self.chol_variational_covar, test_test_covar,
                                                              induc_induc_covar)
            output = GaussianRandomVariable(test_mean, test_covar)
            return output

        else:
            full_covar = gpytorch.add_jitter(full_covar)
            f_prior = GaussianRandomVariable(full_mean, full_covar)
            return f_prior

    def marginal_log_likelihood(self, output, train_y, num_samples=5):
        chol_var_covar = self.chol_variational_covar.triu()

        # Negate each row with a negative diagonal (the Cholesky decomposition
        # of a matrix requires that the diagonal elements be positive).
        chol_var_covar = chol_var_covar.mul(chol_var_covar.diag().sign().unsqueeze(1).expand_as(chol_var_covar).triu())

        _, train_covar = output.representation()
        inducing_output = self.forward(*self.inducing_points)
        inducing_mean = inducing_output.mean()

        train_covar = gpytorch.add_jitter(train_covar)

        log_likelihood = gpytorch.monte_carlo_log_likelihood(self.prior_model.likelihood.log_probability,
                                                             train_y,
                                                             self.variational_mean,
                                                             chol_var_covar,
                                                             train_covar,
                                                             num_samples)

        kl_divergence = gpytorch.mvn_kl_divergence(self.variational_mean,
                                                   chol_var_covar, inducing_mean, train_covar)

        return log_likelihood.squeeze() - kl_divergence
