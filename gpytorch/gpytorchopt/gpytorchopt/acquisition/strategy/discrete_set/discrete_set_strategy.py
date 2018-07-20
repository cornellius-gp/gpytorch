#!/usr/bin/env python3

import torch
from ..acquisition_function_strategy import AcquisitionFunctionStrategy


class DiscreteSetStrategy(AcquisitionFunctionStrategy):
    def __init__(self, acquisition_function, candidate_set):
        super(DiscreteSetStrategy, self).__init__(acquisition_function)
        self._candidate_set = candidate_set

    def pop_index(self, candidate_idx):
        if torch.is_tensor(candidate_idx):
            candidate_idx = candidate_idx.item()
        if candidate_idx == 0:
            self._candidate_set = self._candidate_set[1:, :]
        elif candidate_idx == len(self.candidate_set) - 1:
            self._candidate_set = self._candidate_set[:-1, :]
        else:
            self._candidate_set = torch.cat(
                (self._candidate_set[:candidate_idx, :], self._candidate_set[(candidate_idx + 1) :, :]), 0
            )

    def maximize(self):
        raise NotImplementedError

    @property
    def candidate_set(self):
        return self._candidate_set
