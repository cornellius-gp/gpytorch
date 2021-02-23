#!/usr/bin/env python3

import torch

from ..lazy import BlockDiagLazyTensor, BlockInterleavedLazyTensor, CatLazyTensor, LazyTensor, lazify
from ..utils.broadcasting import _mul_broadcast_shape
from .multivariate_normal import MultivariateNormal


class MultitaskMultivariateNormal(MultivariateNormal):
    """
    Constructs a multi-output multivariate Normal random variable, based on mean and covariance
    Can be multi-output multivariate, or a batch of multi-output multivariate Normal

    Passing a matrix mean corresponds to a multi-output multivariate Normal
    Passing a matrix mean corresponds to a batch of multivariate Normals

    :param torch.Tensor mean:  An `n x t` or batch `b x n x t` matrix of means for the MVN distribution.
    :param ~gpytorch.lazy.LazyTensor covar: An `nt x nt` or batch `b x nt x nt`
        covariance matrix of MVN distribution.
    :param bool validate_args: (default=False) If True, validate `mean` anad `covariance_matrix` arguments.
    :param bool interleaved: (default=True) If True, covariance matrix is interpreted as block-diagonal w.r.t.
        inter-task covariances for each observation. If False, it is interpreted as block-diagonal
        w.r.t. inter-observation covariance for each task.
    """

    def __init__(self, mean, covariance_matrix, validate_args=False, interleaved=True):
        if not torch.is_tensor(mean) and not isinstance(mean, LazyTensor):
            raise RuntimeError("The mean of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, LazyTensor):
            raise RuntimeError("The covariance of a MultitaskMultivariateNormal must be a Tensor or LazyTensor")

        if mean.dim() < 2:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        # Ensure that shapes are broadcasted appropriately across the mean and covariance
        # Means can have singleton dimensions for either the `n` or `t` dimensions
        batch_shape = _mul_broadcast_shape(mean.shape[:-2], covariance_matrix.shape[:-2])
        if mean.shape[-2:].numel() != covariance_matrix.size(-1):
            if covariance_matrix.size(-1) % mean.shape[-2:].numel():
                raise RuntimeError(
                    f"mean shape {mean.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
            elif mean.size(-2) == 1:
                mean = mean.expand(*batch_shape, covariance_matrix.size(-1) // mean.size(-1), mean.size(-1))
            elif mean.size(-1) == 1:
                mean = mean.expand(*batch_shape, mean.size(-2), covariance_matrix.size(-2) // mean.size(-2))
            else:
                raise RuntimeError(
                    f"mean shape {mean.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
        else:
            mean = mean.expand(*batch_shape, *mean.shape[-2:])

        self._output_shape = mean.shape
        # TODO: Instead of transpose / view operations, use a PermutationLazyTensor (see #539) to handle interleaving
        self._interleaved = interleaved
        if self._interleaved:
            mean_mvn = mean.reshape(*mean.shape[:-2], -1)
        else:
            mean_mvn = mean.transpose(-1, -2).reshape(*mean.shape[:-2], -1)
        super().__init__(mean=mean_mvn, covariance_matrix=covariance_matrix, validate_args=validate_args)

    @property
    def base_sample_shape(self):
        """
        Returns the shape of a base sample (without batching) that is used to
        generate a single sample.
        """
        base_sample_shape = self.event_shape
        return base_sample_shape

    @property
    def event_shape(self):
        return self._output_shape[-2:]

    @classmethod
    def from_batch_mvn(cls, batch_mvn, task_dim=-1):
        """
        Reinterprate a batch of multivariate normal distributions as an (independent) multitask multivariate normal
        distribution.

        :param ~gpytorch.distributions.MultivariateNormal batch_mvn: The base MVN distribution.
            (This distribution should have at least one batch dimension).
        :param int task_dim: Which batch dimension should be interpreted as the dimension for the independent tasks.
        :returns: the independent multitask distribution
        :rtype: gpytorch.distributions.MultitaskMultivariateNormal

        Example:
            >>> # model is a gpytorch.models.VariationalGP
            >>> # likelihood is a gpytorch.likelihoods.Likelihood
            >>> mean = torch.randn(4, 2, 3)
            >>> covar_factor = torch.randn(4, 2, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> mvn = gpytorch.distributions.MultivariateNormal(mean, covar)
            >>> print(mvn.event_shape, mvn.batch_shape)
            >>> # torch.Size([3]), torch.Size([4, 2])
            >>>
            >>> mmvn = MultitaskMultivariateNormal.from_batch_mvn(mvn, task_dim=-1)
            >>> print(mmvn.event_shape, mmvn.batch_shape)
            >>> # torch.Size([3, 2]), torch.Size([4])
        """
        orig_task_dim = task_dim
        task_dim = task_dim if task_dim >= 0 else (len(batch_mvn.batch_shape) + task_dim)
        if task_dim < 0 or task_dim > len(batch_mvn.batch_shape):
            raise ValueError(
                f"task_dim of {orig_task_dim} is incompatible with MVN batch shape of {batch_mvn.batch_shape}"
            )

        num_dim = batch_mvn.mean.dim()
        res = cls(
            mean=batch_mvn.mean.permute(*range(0, task_dim), *range(task_dim + 1, num_dim), task_dim),
            covariance_matrix=BlockInterleavedLazyTensor(batch_mvn.lazy_covariance_matrix, block_dim=task_dim),
        )
        return res

    @classmethod
    def from_independent_mvns(cls, mvns):
        """
        Convert an iterable of MVNs into a :obj:`~gpytorch.distributions.MultitaskMultivariateNormal`.
        The resulting distribution will have :attr:`len(mvns)` tasks, and the tasks will be independent.

        :param ~gpytorch.distributions.MultitaskNormal mvn: The base MVN distributions.
        :returns: the independent multitask distribution
        :rtype: gpytorch.distributions.MultitaskMultivariateNormal

        Example:
            >>> # model is a gpytorch.models.VariationalGP
            >>> # likelihood is a gpytorch.likelihoods.Likelihood
            >>> mean = torch.randn(4, 3)
            >>> covar_factor = torch.randn(4, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> mvn1 = gpytorch.distributions.MultivariateNormal(mean, covar)
            >>>
            >>> mean = torch.randn(4, 3)
            >>> covar_factor = torch.randn(4, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> mvn2 = gpytorch.distributions.MultivariateNormal(mean, covar)
            >>>
            >>> mmvn = MultitaskMultivariateNormal.from_independent_mvns([mvn1, mvn2])
            >>> print(mmvn.event_shape, mmvn.batch_shape)
            >>> # torch.Size([3, 2]), torch.Size([4])
        """
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
            *[mvn.lazy_covariance_matrix.unsqueeze(0) for mvn in mvns], dim=0, output_device=mean.device
        )
        covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        return cls(mean=mean, covariance_matrix=covar_lazy, interleaved=False)

    @classmethod
    def from_repeated_mvn(cls, mvn, num_tasks):
        """
        Convert a single MVN into a :obj:`~gpytorch.distributions.MultitaskMultivariateNormal`,
        where each task shares the same mean and covariance.

        :param ~gpytorch.distributions.MultitaskNormal mvn: The base MVN distribution.
        :param int num_tasks: How many tasks to create.
        :returns: the independent multitask distribution
        :rtype: gpytorch.distributions.MultitaskMultivariateNormal

        Example:
            >>> # model is a gpytorch.models.VariationalGP
            >>> # likelihood is a gpytorch.likelihoods.Likelihood
            >>> mean = torch.randn(4, 3)
            >>> covar_factor = torch.randn(4, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> mvn = gpytorch.distributions.MultivariateNormal(mean, covar)
            >>> print(mvn.event_shape, mvn.batch_shape)
            >>> # torch.Size([3]), torch.Size([4])
            >>>
            >>> mmvn = MultitaskMultivariateNormal.from_repeated_mvn(mvn, num_tasks=2)
            >>> print(mmvn.event_shape, mmvn.batch_shape)
            >>> # torch.Size([3, 2]), torch.Size([4])
        """
        return cls.from_batch_mvn(mvn.expand(torch.Size([num_tasks]) + mvn.batch_shape), task_dim=0)

    def expand(self, batch_size):
        new_mean = self.mean.expand(torch.Size(batch_size) + self.mean.shape[-2:])
        new_covar = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-2:])
        res = self.__class__(new_mean, new_covar, interleaved=self._interleaved)
        return res

    def get_base_samples(self, sample_shape=torch.Size()):
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

    def to_data_independent_dist(self):
        """
        Convert a multitask MVN into a batched (non-multitask) MVNs
        The result retains the intertask covariances, but gets rid of the inter-data covariances.
        The resulting distribution will have :attr:`len(mvns)` tasks, and the tasks will be independent.

        :returns: the bached data-independent MVN
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        # Create batch distribution where all data are independent, but the tasks are dependent
        full_covar = self.lazy_covariance_matrix
        num_data, num_tasks = self.mean.shape[-2:]
        data_indices = torch.arange(0, num_data * num_tasks, num_tasks, device=full_covar.device).view(-1, 1, 1)
        task_indices = torch.arange(num_tasks, device=full_covar.device)
        task_covars = full_covar[
            ..., data_indices + task_indices.unsqueeze(-2), data_indices + task_indices.unsqueeze(-1)
        ]
        return MultivariateNormal(self.mean, lazify(task_covars).add_jitter())

    @property
    def variance(self):
        var = super().variance
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = self._output_shape[:-2] + self._output_shape[:-3:-1]
            return var.view(new_shape).transpose(-1, -2).contiguous()
        return var.view(self._output_shape)
