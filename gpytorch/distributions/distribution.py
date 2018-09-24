from __future__ import absolute_import, division, print_function, unicode_literals

from torch.distributions import Distribution as TDistribution


class Distribution(TDistribution):
    @property
    def islazy(self):
        return self._islazy

    def __add__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()
