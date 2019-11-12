#!/usr/bin/env python3

from torch.distributions import Distribution as TDistribution


class _DistributionBase(TDistribution):
    """
    The base class of Distributions. (Same as torch.distribution.Distribution
    or pyro.distribution.Distribution).
    """

    @property
    def islazy(self):
        return self._islazy

    def __add__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()


try:
    # If pyro is installed, add the TorchDistributionMixin
    from pyro.distributions.torch_distribution import TorchDistributionMixin

    class Distribution(_DistributionBase, TorchDistributionMixin):
        pass


except ImportError:

    class Distribution(_DistributionBase):
        pass
