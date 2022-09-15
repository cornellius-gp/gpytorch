#!/usr/bin/env python3

import warnings

import torch
from linear_operator.operators import RootLinearOperator

from ..distributions import MultitaskMultivariateNormal, MultivariateNormal
from ..module import Module
from ._variational_strategy import _VariationalStrategy


class IndependentMultitaskVariationalStrategy(_VariationalStrategy):
    """
    IndependentMultitaskVariationalStrategy wraps an existing
    :obj:`~gpytorch.variational.VariationalStrategy` to produce vector-valued (multi-task)
    output distributions. Each task will be independent of one another.

    The output will either be a :obj:`~gpytorch.distributions.MultitaskMultivariateNormal` distribution
    (if we wish to evaluate all tasks for each input) or a :obj:`~gpytorch.distributions.MultivariateNormal`
    (if we wish to evaluate a single task for each input).

    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: Number of tasks. Should correspond to the batch size of task_dim.
    :param int task_dim: (Default: -1) Which batch dimension is the task dimension
    """

    def __init__(self, base_variational_strategy, num_tasks, task_dim=-1):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.task_dim = task_dim
        self.num_tasks = num_tasks

    @property
    def prior_distribution(self):
        return self.base_variational_strategy.prior_distribution

    @property
    def variational_distribution(self):
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self):
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self):
        return super().kl_divergence().sum(dim=-1)

    def __call__(self, x, task_indices=None, prior=False, **kwargs):
        r"""
        See :class:`LMCVariationalStrategy`.
        """
        function_dist = self.base_variational_strategy(x, prior=prior, **kwargs)

        if task_indices is None:
            # Every data point will get an output for each task
            if (
                self.task_dim > 0
                and self.task_dim > len(function_dist.batch_shape)
                or self.task_dim < 0
                and self.task_dim + len(function_dist.batch_shape) < 0
            ):
                return MultitaskMultivariateNormal.from_repeated_mvn(function_dist, num_tasks=self.num_tasks)
            else:
                function_dist = MultitaskMultivariateNormal.from_batch_mvn(function_dist, task_dim=self.task_dim)
                assert function_dist.event_shape[-1] == self.num_tasks
                return function_dist

        else:
            # Each data point will get a single output corresponding to a single task

            if self.task_dim > 0:
                raise RuntimeError(f"task_dim must be a negative indexed batch dimension: got {self.task_dim}.")
            num_batch = len(function_dist.batch_shape)
            task_dim = num_batch + self.task_dim

            # Create a mask to choose specific task assignment
            shape = list(function_dist.batch_shape + function_dist.event_shape)
            shape[task_dim] = 1
            task_indices = task_indices.expand(shape).squeeze(task_dim)

            # Create a mask to choose specific task assignment
            task_mask = torch.nn.functional.one_hot(task_indices, num_classes=self.num_tasks)
            task_mask = task_mask.permute(*range(0, task_dim), *range(task_dim + 1, num_batch + 1), task_dim)

            mean = (function_dist.mean * task_mask).sum(task_dim)
            covar = (function_dist.lazy_covariance_matrix * RootLinearOperator(task_mask[..., None])).sum(task_dim)
            return MultivariateNormal(mean, covar)


class MultitaskVariationalStrategy(IndependentMultitaskVariationalStrategy):
    """
    IndependentMultitaskVariationalStrategy wraps an existing
    :obj:`~gpytorch.variational.VariationalStrategy`
    to produce a :obj:`~gpytorch.variational.MultitaskMultivariateNormal` distribution.
    All outputs will be independent of one another.

    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: Number of tasks. Should correspond to the batch size of task_dim.
    :param int task_dim: (Default: -1) Which batch dimension is the task dimension
    """

    def __init__(self, base_variational_strategy, num_tasks, task_dim=-1):
        warnings.warn(
            "MultitaskVariationalStrategy has been renamed to IndependentMultitaskVariationalStrategy",
            DeprecationWarning,
        )
        super().__init__(base_variational_strategy, num_tasks, task_dim=-1)
