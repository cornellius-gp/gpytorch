#!/usr/bin/env python3

import warnings

from ..distributions import MultitaskMultivariateNormal
from ..module import Module
from ._variational_strategy import _VariationalStrategy


class IndependentMultitaskVariationalStrategy(_VariationalStrategy):
    """
    IndependentMultitaskVariationalStrategy wraps an existing
    :obj:`~gpytorch.variational.VariationalStrategy`
    to produce a :obj:`~gpytorch.variational.MultitaskMultivariateNormal` distribution.
    All outputs will be independent of one another.

    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: Number of tasks. Should correspond to the batch size of :attr:`task_dim`.
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

    def __call__(self, x, prior=False, **kwargs):
        function_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
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


class MultitaskVariationalStrategy(IndependentMultitaskVariationalStrategy):
    """
    IndependentMultitaskVariationalStrategy wraps an existing
    :obj:`~gpytorch.variational.VariationalStrategy`
    to produce a :obj:`~gpytorch.variational.MultitaskMultivariateNormal` distribution.
    All outputs will be independent of one another.

    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: Number of tasks. Should correspond to the batch size of :attr:`task_dim`.
    :param int task_dim: (Default: -1) Which batch dimension is the task dimension
    """

    def __init__(self, base_variational_strategy, num_tasks, task_dim=-1):
        warnings.warn(
            "MultitaskVariationalStrategy has been renamed to IndependentMultitaskVariationalStrategy",
            DeprecationWarning,
        )
        super().__init__(base_variational_strategy, num_tasks, task_dim=-1)
