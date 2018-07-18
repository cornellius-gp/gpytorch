import torch
from .discrete_set_strategy import DiscreteSetStrategy


class EnsembleDiscreteSetEvaluationStrategy(DiscreteSetStrategy):
    def __init__(self, acquisition_function_list, candidate_set, gp_model_list):
        self.acquisition_function_list = acquisition_function_list
        self._candidate_set = candidate_set
        self.gp_model_list = gp_model_list

    def maximize(self):
        acq_val_array = torch.zeros(len(self.gp_model_list), self._candidate_set.shape[0])
        for i, gp_model in enumerate(self.gp_model_list):
            acq_func = self.acquisition_function_list[i]
            acq_val = acq_func(self._candidate_set)
            acq_val_array[i, :] = acq_val
        mean_acq_val = acq_val_array.mean(dim=0)
        candidate_idx = torch.argmax(mean_acq_val)
        candidate = self._candidate_set[candidate_idx, :]

        self.pop_index(candidate_idx)
        if candidate.ndimension() == 1:
            candidate = candidate.unsqueeze(0)

        return candidate
