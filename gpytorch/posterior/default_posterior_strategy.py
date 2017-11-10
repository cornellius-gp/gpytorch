import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.lazy import LazyVariable
from .posterior_strategy import PosteriorStrategy


class DefaultPosteriorStrategy(PosteriorStrategy):
    def alpha_size(self):
        return self.var.size()[1]

    def exact_posterior_alpha(self, train_mean, train_y):
        res = gpytorch.inv_matmul(self.var, train_y - train_mean)
        return res

    def exact_posterior_mean(self, test_mean, alpha):
        if isinstance(self.var, LazyVariable):
            return self.var.matmul(alpha) + test_mean
        return torch.addmv(test_mean, self.var, alpha)

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar):
        epsilon = Variable(variational_mean.data.new(len(variational_mean),
                           gpytorch.functions.num_trace_samples).normal_())
        samples = chol_var_covar.t().mm(epsilon)
        samples = samples + variational_mean.unsqueeze(1)
        log_likelihood = log_probability_func(samples, train_y)
        return log_likelihood

    def variational_posterior_alpha(self, variational_mean):
        return gpytorch.inv_matmul(self.var, variational_mean)

    def variational_posterior_mean(self, alpha):
        return self.var.matmul(alpha)

    def variational_posterior_covar(self, induc_test_covar, chol_variational_covar,
                                    test_test_covar, induc_induc_covar):
        from ..lazy import NonLazyVariable, MatmulLazyVariable
        # left_factor = K_{mn}K_{nn}^{-1}(S - K_{nn})
        variational_covar = chol_variational_covar.t().matmul(chol_variational_covar)
        left_factor = torch.mm(self.var, gpytorch.inv_matmul(induc_induc_covar,
                                                             variational_covar - induc_induc_covar))
        # right_factor = K_{nn}^{-1}K_{nm}
        right_factor = gpytorch.inv_matmul(induc_induc_covar, induc_test_covar)
        # test_test_covar = K_{mm} + K_{mn}K_{nn}^{-1}(S - K_{nn})K_{nn}^{-1}K_{nm}
        if not isinstance(test_test_covar, LazyVariable):
            test_test_covar = NonLazyVariable(test_test_covar)
        return test_test_covar + MatmulLazyVariable(left_factor, right_factor)
