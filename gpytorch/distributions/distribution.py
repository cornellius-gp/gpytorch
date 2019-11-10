#!/usr/bin/env python3

from torch.distributions import Distribution as TDistribution


class Distribution(TDistribution):
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
