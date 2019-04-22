#!/usr/bin/env python3

import torch
from .multitask_mean import MultitaskMean


class HadamardMultitaskMean(MultitaskMean):
    def forward(self, x, i):
        """
        Evaluate the appropriate mean in self.base_means on each input
        """
        means = [self.base_means[task](x[i == task]) for task in range(self.num_tasks)]

        res = torch.zeros_like(x)
        for task in range(self.num_tasks):
            if len(means[task]) > 0:
                mean = means[task]

                if res[i == task].dim() == means[0].dim() + 1:
                    mean = mean.unsqueeze(dim=-1)

                res[i == task] = mean

        if res.dim() == means[0].dim() + 1:
            res = res.squeeze(dim=-1)

        return res
