from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from torch.distributions.gamma import Gamma
from gpytorch.priors.prior import TorchDistributionPrior


class GammaPrior(TorchDistributionPrior):
    def __init__(self, concentration, rate, log_transform=False, size=None):
        super(GammaPrior, self).__init__()
        if isinstance(concentration, Number) and isinstance(rate, Number):
            concentration = torch.full((size or 1,), float(concentration))
            rate = torch.full((size or 1,), float(rate))
        elif not (torch.is_tensor(concentration) and torch.is_tensor(rate)):
            raise ValueError("concentration and rate must be both either scalars or Tensors")
        elif concentration.shape != rate.shape:
            raise ValueError("concentration and rate must have the same shape")
        elif size is not None:
            raise ValueError("can only set size for scalar concentration and rate")
        self.register_buffer("concentration", concentration.view(-1).clone())
        self.register_buffer("rate", rate.view(-1).clone())
        self._initialize_distributions()
        self._log_transform = log_transform

    def _initialize_distributions(self):
        self._distributions = [
            Gamma(concentration=c, rate=r, validate_args=True) for c, r in zip(self.concentration, self.rate)
        ]

    def is_in_support(self, parameter):
        return bool((parameter > 0).all().item())
