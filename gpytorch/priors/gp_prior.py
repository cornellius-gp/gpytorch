#!/usr/bin/env python3

import torch

from . import Prior


class GaussianProcessPrior(Prior):
    support = torch.distributions.constraints.real
    _validate_args = True
    arg_constraints = {}

    def __init__(self, gp_model, gp_lh, validate_args=False, transform=None):
        super(GaussianProcessPrior, self).__init__(validate_args=validate_args)
        self.gp_model = gp_model
        self.gp_lh = gp_lh
        self._transform = transform

    def log_prob(self, x):
        return self._log_prob(self.transform(x))

    def _log_prob(self, target):
        self.gp_model.set_train_data(targets=target.data, strict=False)

        dist = self.gp_lh(self.gp_model(*self.gp_model.train_inputs))
        return dist.log_prob(target)

    def expand(self, batch_shape):
        return self

    def rsample(self, sample_shape=torch.Size()):
        return self.gp_model(*self.gp_model.train_inputs).rsample(sample_shape)
