from .random_variable import RandomVariable
from torch.autograd import Variable
from gpytorch.lazy import LazyVariable


class GaussianRandomVariable(RandomVariable):
    def __init__(self, mean, covar):
        """
        Constructs a multivariate Gaussian random variable, based on mean and covariance
        Can be multivariate, or a batch of multivariate gaussians

        Passing a vector mean corresponds to a multivariate Gaussian
        Passing a matrix mean corresponds to a batch of multivariate Gaussians

        Params:
        - mean (Variable: vector n or matrix b x n) mean of Gaussian distribution
        - covar (Variable: matrix n x n or batch matrix b x n x n) covariance of Gaussian distribution
        """
        if not isinstance(mean, Variable) and not isinstance(mean, LazyVariable):
            raise RuntimeError('The mean of a GaussianRandomVariable must be a Variable')

        if not isinstance(covar, Variable) and not isinstance(covar, LazyVariable):
            raise RuntimeError('The covariance of a GaussianRandomVariable must be a Variable')

        print(mean.size(), covar.size())
        if not (mean.ndimension() == 1 or mean.ndimension() == 2):
            raise RuntimeError('mean should be a vector or a matrix (batch mode)')

        self._mean = mean
        self._covar = covar

    def covar(self):
        return self._covar

    def mean(self):
        return self._mean

    def representation(self):
        return self._mean, self._covar

    def sample(self, n_samples):
        raise NotImplementedError

    def var(self):
        return self._covar.diag()

    def __len__(self):
        if self._mean.ndimension() == 1:
            return 1
        return self._mean.size(0)
