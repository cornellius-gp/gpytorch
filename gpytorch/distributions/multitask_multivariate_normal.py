#!/usr/bin/env python3

import torch

from ..lazy import BlockDiagLazyTensor, CatLazyTensor, LazyTensor, NonLazyTensor
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

        if mean.ndimension() not in {2, 3}:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        self._output_shape = mean.shape
        super(MultitaskMultivariateNormal, self).__init__(
            mean=mean.view(mean.shape[:-2] + torch.Size([-1])),
            covariance_matrix=covariance_matrix,
            validate_args=validate_args,
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
        if len(mvns[0].batch_shape) > 1:
            raise ValueError("Multiple batch dimensions are not supported in from_independent_mvns.")
        mean = torch.stack([mvn.mean for mvn in mvns], -1)
        # TODO: To do the following efficiently, we don't want to evaluate the
        # covariance matrices. Instead, we want to use the lazies directly in the
        # BlockDiagLazyTensor. This will require implementing a new BatchLazyTensor:
        # https://github.com/cornellius-gp/gpytorch/issues/468
        batch_mode = len(mvns[0].covariance_matrix.shape) == 3
        if batch_mode:
            covar_blocks_lazy = CatLazyTensor(
                *[mvn.lazy_covariance_matrix for mvn in mvns],
                dim=0,
                output_device=mean.device
            )
        else:
            covar_blocks_lazy = NonLazyTensor(
                torch.cat(
                    [mvn.covariance_matrix.unsqueeze(0) for mvn in mvns],
                    dim=0
                )
            )
        covar_lazy = BlockDiagLazyTensor(
            covar_blocks_lazy,
            num_blocks=len(mvns) if batch_mode else None
        )
        return cls(mean=mean, covariance_matrix=covar_lazy)

    def get_base_samples(self, sample_shape=torch.Size()):
        """Get i.i.d. standard Normal samples (to be used with rsample(base_samples=base_samples))"""
        res = super(MultitaskMultivariateNormal, self).get_base_samples(sample_shape)
        res = res.view(*sample_shape, *self._output_shape)
        return res

    def log_prob(self, value):
        return super(MultitaskMultivariateNormal, self).log_prob(value.view(value.shape[:-2] + torch.Size([-1])))

    @property
    def mean(self):
        return super(MultivariateNormal, self).mean.view(self._output_shape)

    @property
    def num_tasks(self):
        return self._output_shape[-1]

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        if base_samples is not None:
            # Make sure that the base samples agree with the distribution
            if tuple(self.mean.size()) != tuple(self.mean.size()[-self.mean.dim() :]):
                raise RuntimeError(
                    "The size of base_samples (minus sample shape dimensions) should agree with the size "
                    "of self.mean. Expected ...{} but got {}".format(self.loc.size(), base_samples.size())
                )

            sample_shape = torch.Size(tuple(base_samples.size(i) for i in range(base_samples.dim() - self.mean.dim())))
            base_samples = base_samples.view(*sample_shape, *self.loc.shape)

        samples = super(MultitaskMultivariateNormal, self).rsample(sample_shape=sample_shape, base_samples=base_samples)
        samples = samples.view(sample_shape + self._output_shape)
        return samples

    @property
    def variance(self):
        return super(MultivariateNormal, self).variance.view(self._output_shape)
