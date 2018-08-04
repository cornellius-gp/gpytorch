from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.means import Mean


class MultitaskMean(Mean):
    def __init__(self, base_means, n_tasks):
        super(MultitaskMean, self).__init__()

        if isinstance(base_means, Mean):
            base_means = [base_means]

        if not isinstance(base_means, list) or (len(base_means) != 1 and len(base_means) != n_tasks):
            raise RuntimeError("base_means should be a list of means of length either 1 or n_tasks")

        if len(base_means) == 1:
            base_means = base_means * n_tasks

        self.base_means = base_means
        self.n_tasks = n_tasks

    def forward(self, input):
        return torch.cat([sub_mean(input).unsqueeze(-1) for sub_mean in self.base_means], dim=-1)
