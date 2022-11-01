#!/usr/bin/env python3

import math
import warnings

import torch
from linear_operator import LinearOperator, to_linear_operator
from linear_operator.operators import BlockDiagLinearOperator, BlockInterleavedLinearOperator, CatLinearOperator

from torch.distributions import MultivariateNormal as TMultivariateNormal
from .multivariate_normal import MultivariateNormal
from torch.distributions.kl import register_kl
from torch.distributions.utils import _standard_normal, lazy_property

from .. import settings
from ..utils.warnings import NumericalWarning


class VeccMultivariateNormal(MultivariateNormal):
    """
    Constructs a block multivariate normal random variable, based on lists of means and covariances.
    Can be multivariate, or a batch of multivariate normals

    Passing a list of vector mean corresponds to a multivariate normal.
    Passing a list of matrix mean corresponds to a batch of multivariate normals.

    :param list mean: List of vectors n or list of matrices b x n means of block conditional mvn distribution.
    :param list covariance_matrix: list of ~linear_operator.operators.LinearOperator or pytorch tensors of
    ... x N X N covariance matrices of block conditional mvn distribution.
    """

    def __init__(self, mean, covariance_matrix, blocks, validate_args=False):
        if not all(torch.is_tensor(this_mean) for this_mean in mean) and \
                not all(isinstance(this_mean, LinearOperator) for this_mean in mean):
            raise RuntimeError("The mean of a VeccMultivariateNormal must be a list of Tensors or LinearOperators")

        if not all(torch.is_tensor(this_cov) for this_cov in covariance_matrix) and \
                not all(isinstance(this_cov, LinearOperator) for this_cov in covariance_matrix):
            raise RuntimeError("The covariance of a VeccMultivariateNormal must be a list of Tensors or LinearOperators")

        self.blocks = blocks

        self.bmvn = [MultivariateNormal(
                        mean=mean[i],
                        covariance_matrix=covariance_matrix[i],
                        validate_args=validate_args)
                     for i in range(len(mean))]

        self._islazy = any([mvn.islazy for mvn in self.bmvn])

    @property
    def islazy(self):
        return self._islazy

    @property
    def event_shape(self):
        return [mvn.loc.shape[-1:] for mvn in self.bmvn]

    @property
    def _unbroadcasted_scale_tril(self):
        raise NotImplementedError

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust):
        raise NotImplementedError

    def add_jitter(self, noise=1e-4):
        raise NotImplementedError

    def expand(self, batch_size):
        raise NotImplementedError

    def _extended_shape(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def confidence_region(self):
        raise NotImplementedError

    @staticmethod
    def _repr_sizes(mean, covariance_matrix):
        raise NotImplementedError

    @lazy_property
    def mean(self):
        if self.islazy:
            return [mvn.loc.to_dense() for mvn in self.bmvn]
        else:
            return [mvn.loc for mvn in self.bmvn]

    @lazy_property
    def covariance_matrix(self):
        raise NotImplementedError

    def get_base_samples(self, sample_shape=torch.Size()):
        raise NotImplementedError

    @property
    def base_sample_shape(self):
        raise NotImplementedError

    @lazy_property
    def lazy_covariance_matrix(self):
        """
        The covariance_matrix, represented as a LinearOperator
        """
        if self.islazy:
            return [mvn._covar for mvn in self.bmvn]
        else:
            return [to_linear_operator(mvn.covariance_matrix) for mvn in self.bmvn]

    def log_prob(self, value):
        return torch.sum(torch.stack([self.bmvn[i].log_prob(value[self.blocks.blocks[i]])
                                      for i in range(len(self.bmvn))]))

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        raise NotImplementedError

    def sample(self, sample_shape=torch.Size(), base_samples=None):
        return torch.cat([mvn.sample() for mvn in self.bmvn])

    def to_data_independent_dist(self):
        raise NotImplementedError

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.bmvn[idx]


@register_kl(VeccMultivariateNormal, VeccMultivariateNormal)
def kl_mvn_mvn(p_dist, q_dist):
    raise NotImplementedError
