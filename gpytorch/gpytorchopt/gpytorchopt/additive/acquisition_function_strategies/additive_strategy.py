import torch
from gpytorchopt.acquisition_function_strategies import AcquisitionFunctionStrategy


class AdditiveStrategy(AcquisitionFunctionStrategy):
    def __init__(self, acquisition_function, candidate_set_factory, additive_structure):
        # acquisition_function_factory(active_kernel) # could return an MVES... or an EI...
        self.acquisition_function = acquisition_function
        # candidate_set_factory is a function that takes in D and returns a D dimensional candidate set
        self.candidate_set_factory = candidate_set_factory
        self.additive_structure = additive_structure

    def maximize(self):
        num_dims = sum([len(group) for group in self.additive_structure])
        best_cand = torch.zeros(1, num_dims)

        for i, group in enumerate(self.additive_structure):
            restricted_candidate_set = self.candidate_set_factory(len(group))
            # restrict the gp_model to kernel[i]
            orig_train_inputs, orig_kernels = self.acquisition_function.gp_model.restrict_kernel(i)

            acq_val = self.acquisition_function(restricted_candidate_set)
            cand_idx = torch.argmax(acq_val)
            restricted_best_cand = restricted_candidate_set[cand_idx, :]
            best_cand[:, group] = restricted_best_cand

            # unrestrict the gp_model
            self.acquisition_function.gp_model.unrestrict_kernel(orig_train_inputs, orig_kernels)

        return best_cand
