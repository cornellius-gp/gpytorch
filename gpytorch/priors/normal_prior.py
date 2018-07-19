from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from torch.distributions.normal import Normal
from gpytorch.priors.prior import TorchDistributionPrior


class NormalPrior(TorchDistributionPrior):
    def __init__(self, loc, scale, log_transform=False, size=None):
        if isinstance(loc, Number) and isinstance(scale, Number):
            loc = torch.full((size or 1,), float(loc))
            scale = torch.full((size or 1,), float(scale))
        elif not (torch.is_tensor(loc) and torch.is_tensor(scale)):
            raise ValueError("loc and scale must be both either scalars or Tensors")
        elif loc.shape != scale.shape:
            raise ValueError("loc and scale must have the same shape")
        elif size is not None:
            raise ValueError("can only set size for scalar loc and scale")
        super(NormalPrior, self).__init__()
        self.register_buffer("loc", loc.view(-1).clone())
        self.register_buffer("scale", scale.view(-1).clone())
        self._initialize_distributions()
        self._log_transform = log_transform

    def _initialize_distributions(self):
        self._distributions = [Normal(loc=lc, scale=sc, validate_args=True) for lc, sc in zip(self.loc, self.scale)]

    def is_in_support(self, parameter):
        return True
