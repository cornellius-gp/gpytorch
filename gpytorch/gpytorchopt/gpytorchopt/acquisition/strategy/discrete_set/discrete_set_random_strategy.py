#!/usr/bin/env python3

import random
import torch
from .discrete_set_strategy import DiscreteSetStrategy


class DiscreteSetRandomStrategy(DiscreteSetStrategy):
    def __init__(self, candidate_set):
        super(DiscreteSetRandomStrategy, self).__init__(None, candidate_set)
        self._candidate_set = candidate_set

    def maximize(self, num_samples=1, remove_candidate=False):
        candidate = torch.Tensor()
        for _ in range(num_samples):
            randind = random.randint(0, self._candidate_set.shape[0] - 1)
            random_candidate = self._candidate_set[randind, :]
            candidate = torch.cat((candidate, random_candidate.unsqueeze(0)))
            if remove_candidate:
                self.pop_index(randind)

        return candidate
