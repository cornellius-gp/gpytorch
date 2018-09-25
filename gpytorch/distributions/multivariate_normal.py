from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.distributions import MultivariateNormal as TMultivariateNormal
from torch.distributions.utils import _standard_normal, lazy_property

from ..lazy import LazyTensor, NonLazyTensor
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
        self._islazy = any(isinstance(arg, LazyTensor) for arg in (mean, covariance_matrix))
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
            super(TMultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=False)
        else:
            super(MultivariateNormal, self).__init__(
                loc=mean, covariance_matrix=covariance_matrix, validate_args=validate_args
            )

    @property
    def _unbroadcasted_scale_tril(self):
        if self.islazy and self.__unbroadcasted_scale_tril is None:
            # cache root decoposition
            self.__unbroadcasted_scale_tril = self.lazy_covariance_matrix.root_decomposition()
        return self.__unbroadcasted_scale_tril

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust):
        if self.islazy:
            raise NotImplementedError("Cannot set _unbroadcasted_scale_tril for lazy MVN distributions")
        else:
            self.__unbroadcasted_scale_tril = ust

    def confidence_region(self):
        """
        Returns 2 standard deviations above and below the mean.

        Returns:
            Tuple[Tensor, Tensor]: pair of tensors of size (b x d) or (d), where
                b is the batch size and d is the dimensionality of the random
                variable. The first (second) Tensor is the lower (upper) end of
                the confidence region.

        """
        std2 = self.stddev.mul_(2)
        mean = self.mean
        return mean.sub(std2), mean.add(std2)

    @lazy_property
    def covariance_matrix(self):
        if self.islazy:
            return self._covar.evaluate()
        else:
            return super(MultivariateNormal, self).covariance_matrix

    def get_base_samples(self, sample_shape=torch.Size()):
        """Get i.i.d. standard Normal samples (to be used with rsample(base_samples=base_samples))"""
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            base_samples = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return base_samples

    @lazy_property
    def lazy_covariance_matrix(self):
        """
        The covariance_matrix, represented as a LazyTensor
        """
        if self.islazy:
            return self._covar
        else:
            return NonLazyTensor(super(MultivariateNormal, self).covariance_matrix)

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        covar = self.lazy_covariance_matrix

        if base_samples is None:
            # Create some samples
            num_samples = sample_shape.numel() or 1

            # Get samples
            res = covar.zero_mean_mvn_samples(num_samples) + self.loc.unsqueeze(0)
            res = res.view(*tuple(sample_shape), *tuple(self.loc.size()))

        else:
            # Make sure that the base samples agree with the distribution
            if tuple(self.loc.size()) != tuple(base_samples.size()[-self.loc.dim() :]):
                raise RuntimeError(
                    "The size of base_samples (minus sample shape dimensions) should agree with the size "
                    "of self.loc. Expected ...{} but got {}".format(self.loc.size(), base_samples.size())
                )

            # Determine what the appropriate sample_shape parameter is
            sample_shape = torch.Size(tuple(base_samples.size(i) for i in range(base_samples.dim() - self.loc.dim())))

            # Reshape samples to be batch_size x num_dim x num_samples
            # or num_bim x num_samples
            base_samples = base_samples.view(-1, *tuple(self.loc.size()))
            base_samples = base_samples.permute(*tuple(i + 1 for i in range(self.loc.dim())), 0)

            # Now reparameterize those base samples
            covar_root = covar.root_decomposition()
            res = covar_root.matmul(base_samples) + self.loc.unsqueeze(-1)

            # Permute and reshape new samples to be original size
            res = res.permute(-1, *tuple(i for i in range(self.loc.dim()))).contiguous()
            res = res.view(*tuple(sample_shape), *tuple(self.loc.size()))

        return res

    def sample(self, sample_shape=torch.Size(), base_samples=None):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)

    @property
    def variance(self):
        if self.islazy:
            # overwrite this since torch MVN uses unbroadcasted_scale_tril for this
            return self.lazy_covariance_matrix.diag().expand(self._batch_shape + self._event_shape)
        else:
            return super(MultivariateNormal, self).variance

    def __add__(self, other):
        if isinstance(other, MultivariateNormal):
            return self.__class__(
                mean=self._mean + other.mean,
                covariance_matrix=(self.lazy_covariance_matrix + other.lazy_covariance_matrix),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.mean + other, self.lazy_covariance_matrix)
        else:
            raise RuntimeError("Unsupported type {} for addition w/ MultivariateNormal".format(type(other)))

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise RuntimeError("Can only multiply by scalars")
        if other == 1:
            return self
        return self.__class__(mean=self.mean * other, covariance_matrix=self.lazy_covariance_matrix * (other ** 2))

    def __truediv__(self, other):
        return self.__mul__(1. / other)
