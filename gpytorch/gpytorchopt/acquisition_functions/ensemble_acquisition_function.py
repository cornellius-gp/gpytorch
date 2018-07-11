from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch


class EnsembleAcquisitionFunction(gpytorch.Module):
    def __init__(self, acq_func, gp_model_list):
        super(EnsembleAcquisitionFunction, self).__init__()
        self.acq_func = acq_func
        self.gp_model_list = gp_model_list

    def forward(self, candidate_set):
        acq_val_array = torch.zeros(len(self.gp_model_list), candidate_set.shape[0])
        for i in range(len(self.gp_model_list)):
            gp_model = self.gp_model_list[i]
            acq = self.acq_func(gp_model)
            acq_val = acq(candidate_set)
            acq_val_array[i, :] = acq_val

        return acq_val_array.mean(dim=0)
