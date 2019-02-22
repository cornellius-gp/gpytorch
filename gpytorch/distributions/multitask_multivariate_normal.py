#!/usr/bin/env python3

import torch

from ..lazy import BlockDiagLazyTensor, CatLazyTensor, LazyTensor
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
            covar (:obj:`torch.tensor` or :obj:`gpytorch.lazy.LazyTensor`): An `nt x nt` or batch `b x nt x nt`
                covariance matrix of MVN distribution.
        """
        if not torch.is_tensor(mean) and not isinstance(mean, LazyTensor):
            raise RuntimeError("The mean of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, LazyTensor):
            raise RuntimeError("The covariance of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if mean.ndimension() < 2:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        self._output_shape = mean.shape
        super().__init__(
            mean=mean.view(*mean.shape[:-2], -1), covariance_matrix=covariance_matrix, validate_args=validate_args
        )

    @property
    def event_shape(self):
        return self._output_shape

    @classmethod
    def from_independent_mvns(cls, mvns):
        if len(mvns) < 2:
            raise ValueError("Must provide at least 2 MVNs to form a MultitaskMultivariateNormal")
        if not all(m.batch_shape == mvns[0].batch_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same batch shape")
        if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same event shape")
        mean = torch.stack([mvn.mean for mvn in mvns], -1)
        covar_blocks_lazy = CatLazyTensor(
            *[mvn.lazy_covariance_matrix.unsqueeze(0) for mvn in mvns], dim=0, output_device=mean.device
        )
        covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        return cls(mean=mean, covariance_matrix=covar_lazy)

    def get_base_samples(self, sample_shape=torch.Size()):
        """Get i.i.d. standard Normal samples (to be used with rsample(base_samples=base_samples))"""
        return super().get_base_samples(sample_shape).view(*sample_shape, *self._output_shape)

    def log_prob(self, value):
        return super().log_prob(value.view(*value.shape[:-2], -1))

    @property
    def mean(self):
        return super().mean.view(self._output_shape)

    @property
    def num_tasks(self):
        return self._output_shape[-1]

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        if base_samples is not None:
            # Make sure that the base samples agree with the distribution
            mean_shape = self.mean.shape
            base_sample_shape = base_samples.shape[-self.mean.ndimension() :]
            if mean_shape != base_sample_shape:
                raise RuntimeError(
                    "The shape of base_samples (minus sample shape dimensions) should agree with the shape "
                    "of self.mean. Expected ...{} but got {}".format(mean_shape, base_sample_shape)
                )
            sample_shape = base_samples.shape[: -self.mean.ndimension()]
            base_samples = base_samples.view(*sample_shape, *self.loc.shape)

        samples = super().rsample(sample_shape=sample_shape, base_samples=base_samples)
        return samples.view(sample_shape + self._output_shape)

    @property
    def variance(self):
        return super().variance.view(self._output_shape)
