#!/usr/bin/env python3

import torch

from ..lazy import BlockDiagLazyTensor, CatLazyTensor, LazyTensor
from .multivariate_normal import MultivariateNormal


class MultitaskMultivariateNormal(MultivariateNormal):
    def __init__(self, mean, covariance_matrix, validate_args=False, interleaved=True):
        """
        Constructs a multi-output multivariate Normal random variable, based on mean and covariance
        Can be multi-output multivariate, or a batch of multi-output multivariate Normal

        Passing a matrix mean corresponds to a multi-output multivariate Normal
        Passing a matrix mean corresponds to a batch of multivariate Normals

        Params:
            mean (:obj:`torch.tensor`): An `n x t` or batch `b x n x t` matrix of means for the MVN distribution.
            covar (:obj:`torch.tensor` or :obj:`gpytorch.lazy.LazyTensor`): An `nt x nt` or batch `b x nt x nt`
                covariance matrix of MVN distribution.
            validate_args (:obj:`bool`): If True, validate `mean` anad `covariance_matrix` arguments.
            interleaved (:obj:`bool`): If True, covariance matrix is interpreted as block-diagonal w.r.t.
                inter-task covariances for each observation. If False, it is interpreted as block-diagonal
                w.r.t. inter-observation covariance for each task.
        """
        if not torch.is_tensor(mean) and not isinstance(mean, LazyTensor):
            raise RuntimeError("The mean of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, LazyTensor):
            raise RuntimeError("The covariance of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if mean.dim() < 2:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        self._output_shape = mean.shape
        # TODO: Instead of transpose / view operations, use a PermutationLazyTensor (see #539) to handle interleaving
        self._interleaved = interleaved
        if self._interleaved:
            mean_mvn = mean.view(*mean.shape[:-2], -1)
        else:
            mean_mvn = mean.transpose(-1, -2).reshape(*mean.shape[:-2], -1)
        super().__init__(mean=mean_mvn, covariance_matrix=covariance_matrix, validate_args=validate_args)

    @property
    def event_shape(self):
        return self._output_shape[-2:]

    @classmethod
    def from_independent_mvns(cls, mvns):
        if len(mvns) < 2:
            raise ValueError("Must provide at least 2 MVNs to form a MultitaskMultivariateNormal")
        if any(isinstance(mvn, MultitaskMultivariateNormal) for mvn in mvns):
            raise ValueError("Cannot accept MultitaskMultivariateNormals")
        if not all(m.batch_shape == mvns[0].batch_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same batch shape")
        if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same event shape")
        mean = torch.stack([mvn.mean for mvn in mvns], -1)
        # TODO: To do the following efficiently, we don't want to evaluate the
        # covariance matrices. Instead, we want to use the lazies directly in the
        # BlockDiagLazyTensor. This will require implementing a new BatchLazyTensor:

        # https://github.com/cornellius-gp/gpytorch/issues/468
        covar_blocks_lazy = CatLazyTensor(
            *[mvn.lazy_covariance_matrix.unsqueeze(0) for mvn in mvns],
            dim=0,
            output_device=mean.device
        )
        covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        return cls(mean=mean, covariance_matrix=covar_lazy, interleaved=False)

    def get_base_samples(self, sample_shape=torch.Size()):
        """Get i.i.d. standard Normal samples (to be used with rsample(base_samples=base_samples))"""
        base_samples = super().get_base_samples(sample_shape)
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = sample_shape + self._output_shape[:-2] + self._output_shape[:-3:-1]
            return base_samples.view(new_shape).transpose(-1, -2).contiguous()
        return base_samples.view(*sample_shape, *self._output_shape)

    def log_prob(self, value):
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = value.shape[:-2] + value.shape[:-3:-1]
            value = value.view(new_shape).transpose(-1, -2).contiguous()
        return super().log_prob(value.view(*value.shape[:-2], -1))

    @property
    def mean(self):
        mean = super().mean
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = self._output_shape[:-2] + self._output_shape[:-3:-1]
            return mean.view(new_shape).transpose(-1, -2).contiguous()
        return mean.view(self._output_shape)

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
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = sample_shape + self._output_shape[:-2] + self._output_shape[:-3:-1]
            return samples.view(new_shape).transpose(-1, -2).contiguous()
        return samples.view(sample_shape + self._output_shape)

    @property
    def variance(self):
        var = super().variance
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = self._output_shape[:-2] + self._output_shape[:-3:-1]
            return var.view(new_shape).transpose(-1, -2).contiguous()
        return var.view(self._output_shape)
