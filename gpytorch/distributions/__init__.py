#!/usr/bin/env python3

from .delta import Delta
from .distribution import Distribution
from .multitask_multivariate_normal import MultitaskMultivariateNormal
from .multivariate_normal import MultivariateNormal

# Get the set of distributions from either PyTorch or Pyro
try:
    # If pyro is installed, use that set of base distributions
    import pyro.distributions as base_distributions
except ImportError:
    # Otherwise, use PyTorch
    import torch.distributions as base_distributions


__all__ = ["Delta", "Distribution", "MultivariateNormal", "MultitaskMultivariateNormal", "base_distributions"]
