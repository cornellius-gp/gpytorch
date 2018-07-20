#!/usr/bin/env python3

import torch
from .additive_strategy import AdditiveStrategy


class DimScanAdditiveStrategy(AdditiveStrategy):
    def __init__(self, acquisition_function, candidate_set_factory, additive_structure):
        super(DimScanAdditiveStrategy, self).__init__(acquisition_function, candidate_set_factory, additive_structure)

        # Set up a base candidate set over the full dimensionality that we will write in to to do dimscan
        total_dim_length = sum(len(group) for group in additive_structure)
        self.base_candidate_set = self.candidate_set_factory(total_dim_length)

    def maximize(self):
        candidate_set = self.base_candidate_set.clone()
        grid_size = candidate_set.shape[0]
        for group in self.additive_structure:
            cand_subset = self.candidate_set_factory(len(group))

            candidate_set[:, group] = cand_subset

            acq_val = self.acquisition_function(candidate_set)
            max_index = torch.argmax(acq_val)
            best_cand = candidate_set[max_index, :]

            candidate_set = best_cand.repeat(grid_size, 1)

        if best_cand.ndimension() == 1:
            best_cand = best_cand.unsqueeze(0)

        return best_cand
