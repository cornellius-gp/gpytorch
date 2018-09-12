from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from ..lazy import LazyVariable, NonLazyVariable
from .multivariate_normal import MultivariateNormal


class MultitaskMultivariateNormal(MultivariateNormal):
    def __init__(self, mean, covariance_matrix, validate_args=False):
        """
        Constructs a multi-output multivariate Normal random variable, based on mean and covariance
        Can be multi-output multivariate, or a batch of multi-output multivariate Normal

        Passing a matrix mean corresponds to a multi-output multivariate Normal
        Passing a matrix mean corresponds to a batch of multivariate Normals

        Params:
            mean (:obj:`torch.tensor`): An `n x t` or batch `b x n x t` matrix of means for the MVN distribution.
            covar (:obj:`torch.tensor` or :obj:`gpytorch.lazy.LazyVariable`): An `nt x nt` or batch `b x nt x nt`
                covariance matrix of MVN distribution.
        """
        super(MultitaskMultivariateNormal, self).__init__(mean, covariance_matrix)
        if not torch.is_tensor(mean) and not isinstance(mean, LazyVariable):
            raise RuntimeError("The mean of a MultitaskMultivariateNormal must be a Tensor or LazyVariable")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, LazyVariable):
            raise RuntimeError("The covariance of a MultitaskMultivariateNormal must be a Tensor or LazyVariable")

        if mean.ndimension() not in {2, 3}:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        if not isinstance(covariance_matrix, LazyVariable):
            covar = NonLazyVariable(covariance_matrix)

        self.loc = mean
        self._covar = covar
        self._nbatch = mean.shape[0] if mean.ndimension() == 3 else None
        self._n = mean.shape[-2]
        self._num_tasks = mean.shape[-1]

    @property
    def n_tasks(self):
        return self._num_tasks

    def covar(self):
        return self._covar

    def mean(self):
        return self.loc

    def representation(self):
        return self.loc, self._covar

    def sample(self, n_samples):
        samples = (
            self._covar.zero_mean_mvn_samples(n_samples)
            .view(self._num_tasks, self._n, n_samples)
            .transpose(0, 1)
            .contiguous()
            .add(self.loc.unsqueeze(-1))
        )
        return samples

    def var(self):
        return self._covar.diag().view(self._n, self._num_tasks)
