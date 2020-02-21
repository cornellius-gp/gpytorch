#!/usr/bin/env python3

import torch

from . import Prior
from .torch_priors import MultivariateNormalPrior

class GaussianProcessPrior(MultivariateNormalPrior):
    def __init__(self, gp_model, validate_args=False, transform=None):
        dist = gp_model(*gp_model.train_inputs)
        super(GaussianProcessPrior, self).__init__(loc=dist.loc, covariance_matrix=dist.covariance_matrix,
                                                   transform=transform, validate_args=validate_args)
        self.gp_model = gp_model
        

    def _log_prob(self, target):
        self.gp_model.set_train_data(targets=target.data, strict=False)

        dist = self.gp_model(*self.gp_model.train_inputs)
        return dist.log_prob(target)
