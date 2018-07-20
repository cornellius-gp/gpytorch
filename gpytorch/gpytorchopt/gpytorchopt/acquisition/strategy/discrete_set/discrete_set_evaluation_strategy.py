#!/usr/bin/env python3

import torch
from .discrete_set_strategy import DiscreteSetStrategy


class DiscreteSetEvaluationStrategy(DiscreteSetStrategy):
    def __init__(self, acquisition_function, candidate_set):
        super(DiscreteSetEvaluationStrategy, self).__init__(acquisition_function, candidate_set)

    def maximize(self):
        # 1. Calls acquisition_function on candidate set
        acq_val = self.acquisition_function(self._candidate_set)

        # 2. Gets the candidate from the candidate set that maximizes acquisition_function
        candidate_idx = torch.argmax(acq_val)
        candidate = self._candidate_set[candidate_idx, :]

        # 3. Removes candidate from candidate set
        self.pop_index(candidate_idx)

        # 4. Return best candidate
        if candidate.ndimension() == 1:
            candidate = candidate.unsqueeze(0)
        return candidate
