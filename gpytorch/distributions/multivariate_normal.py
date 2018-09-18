from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.distributions import MultivariateNormal as TMultivariateNormal
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal, lazy_property

from ..lazy import LazyTensor
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
        self._islazy = any(
            isinstance(arg, LazyTensor) for arg in (mean, covariance_matrix)
        )
        if self._islazy:
            if validate_args:
                # TODO: add argument validation
                raise NotImplementedError()
            self.loc = mean
            self._covar = covariance_matrix
            self.__unbroadcasted_scale_tril = None
            self._validate_args = validate_args
            batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
            # TODO: Integrate argument validation for LazyTensors into torch.distribution validation logic
            super(TMultivariateNormal, self).__init__(
                batch_shape, event_shape, validate_args=False
            )
        else:
            super(MultivariateNormal, self).__init__(
                loc=mean,
                covariance_matrix=covariance_matrix,
                validate_args=validate_args,
            )

    def __add__(self, other):
        if isinstance(other, MultivariateNormal):
            return self.__class__(
                mean=self._mean + other.mean,
                covariance_matrix=self.covariance_matrix + other.covariance_matrix,
            )
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.mean + other, self.covariance_matrix)
        else:
            raise RuntimeError(
                "Unsupported type {} for addition w/ MultivariateNormal".format(
                    type(other)
                )
            )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __truediv__(self, other):
        return self.__mul__(1. / other)

    def __mul__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise RuntimeError("Can only multiply by scalars")
        if other == 1:
            return self
        return self.__class__(
            mean=self.mean * other,
            covariance_matrix=self.covariance_matrix * (other ** 2),
        )

    def representation(self):
        return self.mean, self.covariance_matrix

    @property
    def _unbroadcasted_scale_tril(self):
        if self.islazy and self.__unbroadcasted_scale_tril is None:
            # cache root decoposition
            self.__unbroadcasted_scale_tril = (
                self.covariance_matrix.root_decomposition()
            )
        return self.__unbroadcasted_scale_tril

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust):
        if self.islazy:
            raise NotImplementedError(
                "Cannot set _unbroadcasted_scale_tril for lazy MVN distributions"
            )
        else:
            self.__unbroadcasted_scale_tril = ust

    def get_base_samples(self, sample_shape=torch.Size()):
        """Get i.i.d. standard Normal samples (to be used with correlate_base_samples)"""
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            base_samples = _standard_normal(
                shape, dtype=self.loc.dtype, device=self.loc.device
            )
        return base_samples

    def correlate_base_samples(self, base_samples):
        """Correlate i.i.d. standard Normal samples using the root decomposition
        of the covariance matrix"""
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, base_samples)

    @property
    def variance(self):
        if self.islazy:
            # overwrite this since torch MVN uses unbroadcasted_scale_tril for this
            return self.covariance_matrix.diag().expand(
                self._batch_shape + self._event_shape
            )
        else:
            return super(MultivariateNormal, self).variance

    @lazy_property
    def covariance_matrix(self):
        if self.islazy:
            return self._covar
        else:
            return super(MultivariateNormal, self).covariance_matrix
