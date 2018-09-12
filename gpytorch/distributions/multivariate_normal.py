from __future__ import absolute_import, division, print_function, unicode_literals

import math

import torch
from torch.distributions import MultivariateNormal as TMultivariateNormal
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import lazy_property

from ..lazy import LazyVariable
from .distribution import Distribution


class MultivariateNormal(TMultivariateNormal, Distribution):
    """
    Constructs a multivariate Normal random variable, based on mean and covariance
    Can be multivariate, or a batch of multivariate Normals

    Passing a vector mean corresponds to a multivariate Normal
    Passing a matrix mean corresponds to a batch of multivariate Normals

    Args:
        mean (Tensor): vector n or matrix b x n mean of MVN distribution
        covar (Tensor): matrix n x n or batch matrix b x n x n covariance of
            MVN distribution
    """

    def __init__(self, mean, covariance_matrix, validate_args=False):
        self._islazy = any(isinstance(arg, LazyVariable) for arg in (mean, covariance_matrix))
        if self._islazy:
            if validate_args:
                # TODO: add argument validation
                raise NotImplementedError()
            self.loc = mean
            self._covar = covariance_matrix
            self._validate_args = validate_args
            batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
            # TODO: Integrate argument validation for LazyVariables into torch.distribution validation logic
            super(TMultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=False)
            super(MultivariateNormal, self).__init__(
                loc=mean, covariance_matrix=covariance_matrix, validate_args=validate_args
            )

    def __add__(self, other):
        if isinstance(other, MultivariateNormal):
            return MultivariateNormal(
                mean=self._mean + other.mean, covariance_matrix=self.covariance_matrix + other.covariance_matrix
            )
        elif isinstance(other, int) or isinstance(other, float):
            return MultivariateNormal(self.mean + other, self.covariance_matrix)
        else:
            raise RuntimeError("Unsupported type {} for addition w/ MultivariateNormal".format(type(other)))

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __div__(self, other):
        return self.__mul__(1. / other)

    def __mul__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise RuntimeError("Can only multiply by scalars")
        return self.__class__(mean=self.mean * other, covariance_matrix=self.covariance_matrix * (other ** 2))

    @lazy_property
    def scale_tril(self):
        if self._islazy:
            raise NotImplementedError("scale_tril not implemented for lazy MultivariateNormal")
        else:
            return super(MultivariateNormal, self).scale_tril

    @lazy_property
    def covariance_matrix(self):
        if self._islazy:
            return self._covar
        else:
            return super(MultivariateNormal, self).covariance_matrix

    @lazy_property
    def precision_matrix(self):
        if self._islazy:
            raise NotImplementedError("precision_matrix not implemented for lazy MultivariateNormal")
        else:
            return super(MultivariateNormal, self).precision_matrix

    def rsample(self, sample_shape=torch.Size()):
        if self._islazy:
            shape = self._extended_shape(sample_shape)
            eps = self.loc.new_empty(shape).normal_()
            # this will fail, rewrite using LazyVariable code
            return self.loc + _batch_mv(self._covar.root_decomposition(), eps)
        else:
            return super(MultivariateNormal, self).rsample(sample_shape=sample_shape)

    def log_prob(self, value):
        if self._islazy:
            if self._validate_args:
                self._validate_sample(value)
            diff = value - self.loc
            # re-write this for LazyVariables
            # M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
            # half_log_det = _batch_diag(self._unbroadcasted_scale_tril).log().sum(-1)
            return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det
        else:
            return super(MultivariateNormal, self).log_prob(value)

    def entropy(self):
        if self._islazy:
            # re-write this for LazyVariables
            # half_log_det = _batch_diag(self._unbroadcasted_scale_tril).log().sum(-1)
            H = 0.5 * self._event_shape[0] * (1 + math.log(2 * math.pi)) + half_log_det
            if len(self._batch_shape) == 0:
                return H
            else:
                return H.expand(self._batch_shape)
        else:
            return super(MultivariateNormal, self).entropy()
