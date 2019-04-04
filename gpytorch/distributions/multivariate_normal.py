#!/usr/bin/env python3

import math

import torch
from torch.distributions import MultivariateNormal as TMultivariateNormal
from torch.distributions.kl import register_kl
from torch.distributions.utils import _standard_normal, lazy_property

from .. import settings
from ..lazy import LazyTensor, lazify
from .distribution import Distribution
from ..utils.broadcasting import _mul_broadcast_shape


class _MultivariateNormalBase(TMultivariateNormal, Distribution):
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
        self._islazy = isinstance(mean, LazyTensor) or isinstance(covariance_matrix, LazyTensor)
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
            super().__init__(loc=mean, covariance_matrix=covariance_matrix, validate_args=validate_args)

    @property
    def _unbroadcasted_scale_tril(self):
        if self.islazy and self.__unbroadcasted_scale_tril is None:
            # cache root decoposition
            with settings.fast_computations(covar_root_decomposition=False):
                ust = self.lazy_covariance_matrix.root_decomposition().root.evaluate()
            self.__unbroadcasted_scale_tril = ust
        return self.__unbroadcasted_scale_tril

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust):
        if self.islazy:
            raise NotImplementedError("Cannot set _unbroadcasted_scale_tril for lazy MVN distributions")
        else:
            self.__unbroadcasted_scale_tril = ust

    def expand(self, batch_size):
        new_loc = self.loc.expand(torch.Size(batch_size) + self.loc.shape[-1:])
        new_covar = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-2:])
        res = self.__class__(new_loc, new_covar)
        return res

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
            return super().covariance_matrix

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
            return lazify(super().covariance_matrix)

    def log_prob(self, value):
        if settings.fast_computations.log_prob.off():
            return super().log_prob(value)

        if self._validate_args:
            self._validate_sample(value)

        mean, covar = self.loc, self.lazy_covariance_matrix
        diff = value - mean

        # Repeat the covar to match the batch shape of diff
        if diff.shape[:-1] != covar.batch_shape:
            if len(diff.shape[:-1]) < len(covar.batch_shape):
                diff = diff.expand(covar.shape[:-1])
            else:
                padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - covar.dim())), *covar.batch_shape)
                covar = covar.repeat(
                    *(diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)),
                    1, 1
                )

        # Get log determininat and first part of quadratic form
        inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)

        res = -0.5 * sum([inv_quad, logdet, diff.size(-1) * math.log(2 * math.pi)])
        return res

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        covar = self.lazy_covariance_matrix
        if base_samples is None:
            # Create some samples
            num_samples = sample_shape.numel() or 1

            # Get samples
            res = covar.zero_mean_mvn_samples(num_samples) + self.loc.unsqueeze(0)
            res = res.view(sample_shape + self.loc.shape)

        else:
            # Make sure that the base samples agree with the distribution
            if self.loc.shape != base_samples.shape[-self.loc.dim() :]:
                raise RuntimeError(
                    "The size of base_samples (minus sample shape dimensions) should agree with the size "
                    "of self.loc. Expected ...{} but got {}".format(self.loc.shape, base_samples.shape)
                )

            # Determine what the appropriate sample_shape parameter is
            sample_shape = base_samples.shape[: base_samples.dim() - self.loc.dim()]

            # Reshape samples to be batch_size x num_dim x num_samples
            # or num_bim x num_samples
            base_samples = base_samples.view(-1, *self.loc.shape)
            base_samples = base_samples.permute(*tuple(range(1, self.loc.dim() + 1)), 0)

            # Now reparameterize those base samples
            covar_root = covar.root_decomposition().root
            # If necessary, adjust base_samples for rank of root decomposition
            if covar_root.shape[-1] < base_samples.shape[-2]:
                base_samples = base_samples[..., : covar_root.shape[-1], :]
            elif covar_root.shape[-1] > base_samples.shape[-2]:
                raise RuntimeError("Incompatible dimension of `base_samples`")
            res = covar_root.matmul(base_samples) + self.loc.unsqueeze(-1)

            # Permute and reshape new samples to be original size
            res = res.permute(-1, *tuple(range(self.loc.dim()))).contiguous()
            res = res.view(sample_shape + self.loc.shape)

        return res

    def sample(self, sample_shape=torch.Size(), base_samples=None):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)

    @property
    def variance(self):
        if self.islazy:
            # overwrite this since torch MVN uses unbroadcasted_scale_tril for this
            if len(self._batch_shape) == 2:
                return self.lazy_covariance_matrix.diag().unsqueeze(-1).expand(self._batch_shape + self._event_shape)
            else:
                return self.lazy_covariance_matrix.diag().expand(self._batch_shape + self._event_shape)
        else:
            return super().variance

    def __add__(self, other):
        if isinstance(other, _MultivariateNormalBase):
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
        return self.__mul__(1.0 / other)


try:
    # If pyro is installed, add the TorchDistributionMixin
    from pyro.distributions.torch_distribution import TorchDistributionMixin

    class MultivariateNormal(_MultivariateNormalBase, TorchDistributionMixin):
        pass


except ImportError:

    class MultivariateNormal(_MultivariateNormalBase):
        pass


@register_kl(MultivariateNormal, MultivariateNormal)
def kl_mvn_mvn(p_dist, q_dist):
    output_shape = _mul_broadcast_shape(p_dist.batch_shape, q_dist.batch_shape)
    if output_shape != p_dist.batch_shape:
        p_dist = p_dist.expand(output_shape)
    if output_shape != q_dist.batch_shape:
        q_dist = q_dist.expand(output_shape)

    q_mean = q_dist.loc
    q_covar = q_dist.lazy_covariance_matrix

    p_mean = p_dist.loc
    p_covar = p_dist.lazy_covariance_matrix
    root_p_covar = p_covar.root_decomposition().root.evaluate()

    mean_diffs = p_mean - q_mean
    if isinstance(root_p_covar, LazyTensor):
        # right now this just catches if root_p_covar is a DiagLazyTensor,
        # but we may want to be smarter about this in the future
        root_p_covar = root_p_covar.evaluate()
    inv_quad_rhs = torch.cat([mean_diffs.unsqueeze(-1), root_p_covar], -1)
    logdet_p_covar = p_covar.logdet()
    trace_plus_inv_quad_form, logdet_q_covar = q_covar.inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=True)

    # Compute the KL Divergence.
    res = 0.5 * sum([logdet_q_covar, logdet_p_covar.mul(-1), trace_plus_inv_quad_form, -float(mean_diffs.size(-1))])
    return res
