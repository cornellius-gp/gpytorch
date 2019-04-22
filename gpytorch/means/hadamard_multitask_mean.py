#!/usr/bin/env python3

import torch
from .multitask_mean import MultitaskMean


class HadamardMultitaskMean(MultitaskMean):
    def forward(self, x, i):
        """
        Evaluate the appropriate mean in self.base_means on each input
        """
        means = [self.base_means[task](x[i == task]) for task in range(self.num_tasks)]

        res = torch.zeros_like(torch.cat(means))
        for task in range(self.num_tasks):
            res[(i == task).flatten()] = means[task]

        return res
