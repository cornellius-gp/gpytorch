from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
from .random_variable import RandomVariable
from ..lazy import LazyVariable, NonLazyVariable


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
        super(GaussianRandomVariable, self).__init__(mean, covar)
        if not isinstance(mean, Variable) and not isinstance(mean, LazyVariable):
            raise RuntimeError("The mean of a GaussianRandomVariable must be a Variable")

        if not isinstance(covar, Variable) and not isinstance(covar, LazyVariable):
            raise RuntimeError("The covariance of a GaussianRandomVariable must be a Variable")

        if not (mean.ndimension() == 1 or mean.ndimension() == 2):
            raise RuntimeError("mean should be a vector or a matrix (batch mode)")

        if not isinstance(covar, LazyVariable):
            covar = NonLazyVariable(covar)

        self._mean = mean
        self._covar = covar

    def covar(self):
        return self._covar

    def mean(self):
        return self._mean

    def representation(self):
        return self._mean, self._covar

    def sample(self, n_samples):
        covar = self.covar()
        return covar.zero_mean_mvn_samples(n_samples) + self._mean.unsqueeze(-1)

    def var(self):
        return self._covar.diag()

    def __len__(self):
        if self._mean.ndimension() == 1:
            return 1
        return self._mean.size(0)

    def __add__(self, other):
        if isinstance(other, GaussianRandomVariable):
            return GaussianRandomVariable(self._mean + other.mean(), self._covar + other.covar())
        elif isinstance(other, int) or isinstance(other, float):
            return GaussianRandomVariable(self._mean + other, self._covar)
        else:
            raise RuntimeError("Unsupported type for addition w/ Gaussian random variables")

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __div__(self, other):
        return self.__mul__(1. / other)

    def __mul__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise RuntimeError("Can only multiply by scalars")
        return GaussianRandomVariable(self._mean * other, self._covar * (other ** 2))
