import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.random_variables import GaussianRandomVariable
from .gp_posterior import _GPPosterior


class _ExactGPPosterior(_GPPosterior):
    def __init__(self, prior_model, train_xs=None, train_y=None):
        super(_ExactGPPosterior, self).__init__(prior_model.likelihood)
        self.prior_model = prior_model

        # Buffers for conditioning on data
        if train_xs is not None and train_y is not None:
            self.update_data(train_xs, train_y)

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
        return self

    def forward(self, *inputs, **params):
        n = len(self.train_xs[0]) if hasattr(self, 'train_xs') else 0

        # Compute mean and full data (train/test) covar
        if n:
            train_x_vars = [Variable(train_x) for train_x in self.train_xs]
            full_inputs = [torch.cat([train_x_var, input]) for train_x_var, input in zip(train_x_vars, inputs)]
        else:
            full_inputs = inputs
        gaussian_rv_output, log_noise = self.prior_model.forward(*full_inputs, **params)
        full_mean, full_covar = gaussian_rv_output.representation()

        # Get mean/covar components
        test_mean = full_mean[n:]
        test_test_covar = full_covar[n:, n:]

        # If there's data, use it
        if n:
            train_mean = full_mean[:n]
            train_train_covar = gpytorch.add_diag(full_covar[:n, :n], log_noise.exp())
            train_test_covar = full_covar[:n, n:]
            test_train_covar = full_covar[n:, :n]

            if hasattr(self, 'alpha') and self.alpha is not None:
                GRV, alpha = gpytorch._exact_predict(test_mean, test_test_covar, self.train_y, train_mean,
                                                     train_train_covar, train_test_covar, test_train_covar, self.alpha)
            else:
                GRV, alpha = gpytorch._exact_predict(test_mean, test_test_covar, self.train_y, train_mean,
                                                     train_train_covar, train_test_covar, test_train_covar)

            self.alpha = alpha
        else:
            GRV = GaussianRandomVariable(test_mean, test_test_covar)

        return GRV, log_noise

    def marginal_log_likelihood(self, output, train_y):
        mean, covar = output.representation()
        return gpytorch.exact_gp_marginal_log_likelihood(covar, train_y - mean)
