import gpytorch
from .random_variable import RandomVariable
from torch.autograd import Variable


class GaussianRandomVariable(RandomVariable):
    def __init__(self, mean, var):
        if not isinstance(mean, Variable):
            raise RuntimeError('The mean of a GaussianRandomVariable must be a Variable')

        if not isinstance(var, Variable):
            raise RuntimeError('The mean of a GaussianRandomVariable must be a Variable')

        self._mean = mean
        self._var = var

    def __repr__(self):
        return repr(self.representation())

    def __len__(self):
        return self._mean.__len__()

    def log_probability(self, x):
        return gpytorch.exact_gp_marginal_log_likelihood(self.covar(), x - self.mean())

    def representation(self):
        return self._mean, self._var

    def mean(self):
        return self._mean

    def covar(self):
        return self._var

    def var(self):
        return self.covar().diag()
