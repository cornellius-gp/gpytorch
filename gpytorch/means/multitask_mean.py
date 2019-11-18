#!/usr/bin/env python3

from copy import deepcopy

import torch
from torch.nn import ModuleList

from .mean import Mean


class MultitaskMean(Mean):
    """
    Convenience :class:`gpytorch.means.Mean` implementation for defining a different mean for each task in a multitask
    model. Expects a list of `num_tasks` different mean functions, each of which is applied to the given data in
    :func:`~gpytorch.means.MultitaskMean.forward` and returned as an `n x t` matrix of means, one for each task.
    """

    def __init__(self, base_means, num_tasks):
        """
        Args:
            base_means (:obj:`list` or :obj:`gpytorch.means.Mean`): If a list, each mean is applied to the data.
                If a single mean (or a list containing a single mean), that mean is copied `t` times.
            num_tasks (int): Number of tasks. If base_means is a list, this should equal its length.
        """
        super(MultitaskMean, self).__init__()

        if isinstance(base_means, Mean):
            base_means = [base_means]

        if not isinstance(base_means, list) or (len(base_means) != 1 and len(base_means) != num_tasks):
            raise RuntimeError("base_means should be a list of means of length either 1 or num_tasks")

        if len(base_means) == 1:
            base_means = base_means + [deepcopy(base_means[0]) for i in range(num_tasks - 1)]

        self.base_means = ModuleList(base_means)
        self.num_tasks = num_tasks

    def forward(self, input):
        """
        Evaluate each mean in self.base_means on the input data, and return as an `n x t` matrix of means.
        """
        return torch.cat([sub_mean(input).unsqueeze(-1) for sub_mean in self.base_means], dim=-1)
