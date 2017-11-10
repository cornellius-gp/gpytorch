import gpytorch
from torch.autograd import Variable
from .posterior_strategy import PosteriorStrategy


class InterpolatedPosteriorStrategy(PosteriorStrategy):
    def __init__(self, var, grid, interp_left, interp_right):
        """
        Assumes that var is represented by interp_left * grid * interp_right^T

        Args:
        - var (LazyVariable) - the variable to define the PosteriorStrategy for
        - grid (LazyVariable) - represents the grid matrix
        - interp_left (Sparse Tensor) - the left interpolation matrix of the grid
        - interp_right (Sparse Tensor) - the right interpolation matrix of the grid
        """
        super(InterpolatedPosteriorStrategy, self).__init__(var)
        self.grid = grid
        self.interp_left = interp_left
        self.interp_right = interp_right

    def alpha_size(self):
        return self.grid.size()[1]

    def exact_posterior_alpha(self, train_mean, train_y):
        train_residual = (train_y - train_mean).unsqueeze(1)
        gpytorch.functions.max_cg_iterations *= 10
        alpha = self.var.inv_matmul(train_residual)
        gpytorch.functions.max_cg_iterations /= 10
        alpha = gpytorch.dsmm(self.interp_right.t(), alpha)
        alpha = self.grid.matmul(alpha)
        return alpha.squeeze()

    def exact_posterior_mean(self, test_mean, alpha):
        alpha = alpha.unsqueeze(1)
        return test_mean.add(gpytorch.dsmm(self.interp_left, alpha).squeeze())

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar):
        epsilon = Variable(variational_mean.data.new(variational_mean.size(0),
                           gpytorch.functions.num_trace_samples).normal_())
        samples = chol_var_covar.mm(epsilon)
        samples = samples + variational_mean.unsqueeze(1).expand_as(samples)
        samples = gpytorch.dsmm(self.interp_left, samples)
        log_likelihood = log_probability_func(samples, train_y)

        return log_likelihood

    def variational_posterior_alpha(self, variational_mean):
        return variational_mean.add(0)  # Trick to ensure that we're not returning a Paremeter

    def variational_posterior_mean(self, alpha):
        return gpytorch.dsmm(self.interp_left, alpha.unsqueeze(1)).squeeze()

    def variational_posterior_covar(self, induc_test_covar, chol_variational_covar,
                                    test_test_covar, induc_induc_covar):
        from ..lazy import MatmulLazyVariable
        covar_right = gpytorch.dsmm(self.interp_left, chol_variational_covar.t()).t()
        covar_left = gpytorch.dsmm(self.interp_left, chol_variational_covar.t())
        return MatmulLazyVariable(covar_left, covar_right)
