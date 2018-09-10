from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .gaussian_random_variable import GaussianRandomVariable
from ..lazy import LazyVariable, NonLazyVariable


class MultitaskGaussianRandomVariable(GaussianRandomVariable):
    def __init__(self, mean, covar):
        """
        Constructs a multi-output multivariate Gaussian random variable, based on mean and covariance
        Can be multi-output multivariate, or a batch of multi-output multivariate Gaussians

        Passing a matrix mean corresponds to a multi-output multivariate Gaussian
        Passing a matrix mean corresponds to a batch of multivariate Gaussians

        Params:
            mean (:obj:`torch.tensor`): An `n x t` or batch `b x n x t` matrix of means for the Gaussian distribution.
            covar (:obj:`torch.tensor` or :obj:`gpytorch.lazy.LazyVariable`): An `nt x nt` or batch `b x nt x nt`
                covariance matrix of Gaussian distribution.
        """
        super(MultitaskGaussianRandomVariable, self).__init__(mean, covar)
        if not torch.is_tensor(mean) and not isinstance(mean, LazyVariable):
            raise RuntimeError("The mean of a GaussianRandomVariable must be a Tensor or LazyVariable")

        if not torch.is_tensor(covar) and not isinstance(covar, LazyVariable):
            raise RuntimeError("The covariance of a GaussianRandomVariable must be a Tensor or LazyVariable")

        if mean.ndimension() not in {2, 3}:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        if not isinstance(covar, LazyVariable):
            covar = NonLazyVariable(covar)

        self._mean = mean
        self._covar = covar
        self._nbatch = mean.shape[0] if mean.ndimension() == 3 else None
        self._n = mean.shape[-2]
        self._t = mean.shape[-1]

    @property
    def n_tasks(self):
        return self._t

    def covar(self):
        return self._covar

    def mean(self):
        return self._mean

    def representation(self):
        return self._mean, self._covar

    def sample(self, n_samples):
        samples = (
            self._covar.zero_mean_mvn_samples(n_samples)
            .view(self._t, self._n, n_samples)
            .transpose(0, 1)
            .contiguous()
            .add(self._mean.unsqueeze(-1))
        )
        return samples

    def correlate_base_samples(self, base_samples):
        corr_samples = self._covar.correlate_mvn_base_samples(base_samples)
        samples = (
            corr_samples
            .view(self._t, self._n, base_samples.shape[-1])
            .transpose(0, 1)
            .contiguous()
            .add(self._mean.unsqueeze(-1))
        )
        return samples

    def var(self):
        return self._covar.diag().view(self._n, self._t)
