#!/usr/bin/env python3

from .distribution import Distribution
from .multivariate_normal import MultivariateNormal
from .multitask_multivariate_normal import MultitaskMultivariateNormal

# Get the set of distributions from either PyTorch or Pyro
try:
    # If pyro is installed, use that set of base distributions
    import pyro.distributions as base_distributions
except ImportError:
    # Otherwise, use PyTorch
    import torch.distributions as base_distributions


__all__ = ["Distribution", "MultivariateNormal", "MultitaskMultivariateNormal", "base_distributions"]
