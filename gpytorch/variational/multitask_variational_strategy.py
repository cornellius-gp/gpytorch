#!/usr/bin/env python3

from ._variational_strategy import _VariationalStrategy
from ..module import Module
from ..utils.memoize import cached
from ..distributions import MultitaskMultivariateNormal
from ..lazy import BlockDiagLazyTensor


class MultitaskVariationalStrategy(_VariationalStrategy):
    """
    MultitaskVariationalStrategy wraps an existing :obj:`~gpytorch.variational.VariationalStrategy`
    to product a :obj:`~gpytorch.variational.MultitaskMultivariateNormal` distribution.
    This is useful for multi-output variational models.

    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    Args:
        :attr:`base_variational_strategy` (:obj:`~gpytorch.variational.VariationalStrategy`):
            Base variational strategy
        :attr:`task_dim` (int, default=-2):
            Which batch dimension is the task dimension
    """
    def __init__(self, base_variational_strategy, task_dim=-2):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.task_dim = task_dim

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        return self.base_variational_strategy.prior_distribution

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self):
        return self.base_variational_strategy.variational_params_initialized

    def forward(self, inputs):
        function_dist = self.base_variational_strategy(inputs)
        num_dim = function_dist.mean.dim()
        block_dim = (num_dim + self.task_dim) if self.task_dim < 0 else self.task_dim
        function_dist = MultitaskMultivariateNormal(
            mean=function_dist.mean.permute(*range(0, block_dim), *range(block_dim + 1, num_dim), block_dim),
            covariance_matrix=BlockDiagLazyTensor(function_dist.lazy_covariance_matrix, block_dim=block_dim),
            interleaved=True,
        )
        return function_dist

    def kl_divergence(self):
        return super().kl_divergence().sum(dim=-1)

    def __call__(self, x):
        # Delete previously cached items from the training distribution
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()
        return Module.__call__(self, x)
