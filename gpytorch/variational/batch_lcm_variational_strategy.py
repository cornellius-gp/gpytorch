#!/usr/bin/env python3

import torch

from ..distributions import MultitaskMultivariateNormal
from ..lazy import KroneckerProductLazyTensor
from ..module import Module
from ._variational_strategy import _VariationalStrategy


class BatchLCMVariationalStrategy(_VariationalStrategy):
    """
    MultitaskVariationalStrategy wraps an existing :obj:`~gpytorch.variational.VariationalStrategy`
    to product a :obj:`~gpytorch.variational.MultitaskMultivariateNormal` distribution.
    This is useful for multi-output variational models.

    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int task_dim: (default=-1) Which batch dimension is the task dimension
    """

    def __init__(
        self, base_variational_strategy, num_tasks, num_functions=0, num_groups=None, function_dim=0, group_dim=1
    ):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.num_tasks = num_tasks
        self.num_functions = num_functions or num_tasks
        self.num_groups = num_groups

        assert function_dim < group_dim
        self.batch_shape = self.base_variational_strategy._variational_distribution.batch_shape
        self.function_dim = function_dim if function_dim < 0 else (function_dim - len(self.batch_shape))
        self.group_dim = group_dim if group_dim < 0 else (group_dim - len(self.batch_shape))

        if not (self.batch_shape[self.group_dim] == num_groups or self.batch_shape[self.group_dim] == 1):
            raise RuntimeError(
                f"Mismatch in num_groups: got a variational distribution of batch shape {self.batch_shape}, "
                f"expected the gruop dim {self.group_dim} to be {self.num_groups}."
            )

        # Ensure the number of tasks/groups is equal to what we have in the variational distribution
        if not (self.batch_shape[self.function_dim] == num_functions or self.batch_shape[self.function_dim] == 1):
            raise RuntimeError(
                f"Mismatch in num_functions: got a variational distribution of batch shape {self.batch_shape}, "
                f"expected the function dim {self.function_dim} to be {self.num_functions}."
            )
        self.batch_shape = list(self.batch_shape)
        del self.batch_shape[self.function_dim]
        self.batch_shape = torch.Size(self.batch_shape)

        # LCM coefficients
        raw_lcm_coefficients = torch.randn(
            *self.batch_shape,
            self.num_tasks,
            self.num_functions,
        ).tril_()
        self.register_parameter("raw_lcm_coefficients", torch.nn.Parameter(raw_lcm_coefficients))

    @property
    def lcm_coefficients(self):
        return self.raw_lcm_coefficients.tril()

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
        if self.group_dim is not None:
            return super().kl_divergence().sum(dim=[self.function_dim, self.group_dim])
        else:
            return super().kl_divergence().sum(dim=[self.function_dim])

    def __call__(self, x, prior=False):
        function_dist = self.base_variational_strategy(x, prior=prior)
        lcm_coefficients = self.lcm_coefficients
        num_batch = len(function_dist.batch_shape)
        num_dim = num_batch + len(function_dist.event_shape)
        function_dim = num_batch + self.function_dim
        group_dim = num_batch + self.group_dim

        # Mean
        mean = function_dist.mean.permute(*range(0, function_dim), *range(function_dim + 1, num_dim), function_dim)
        mean = mean @ lcm_coefficients.transpose(-1, -2)
        mean = mean.sum(group_dim - 1)

        # Covar
        covar = function_dist.lazy_covariance_matrix
        covar = covar.sum(function_dim)
        lcm_factor = lcm_coefficients @ lcm_coefficients.transpose(-1, -2)
        lcm_factor = lcm_factor.expand(*covar.batch_shape, *lcm_factor.shape[-2:])
        covar = KroneckerProductLazyTensor(covar, lcm_factor)
        covar = covar.sum(group_dim - 1)  # - 1 because we summed over the function_dim

        # Done!
        function_dist = MultitaskMultivariateNormal(mean, covar)
        return function_dist
