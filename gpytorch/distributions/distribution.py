from __future__ import absolute_import, division, print_function, unicode_literals

from torch.distributions import Distribution as TDistribution


class Distribution(TDistribution):
    def confidence_region(self):
        """
        Returns 2 standard deviations above and below the mean.

        Returns:
            Tuple[Tensor, Tensor]: pair of tensors of size (b x d) or (d), where
                b is the batch size and d is the dimensionality of the random
                variable. The first (second) Tensor is the lower (upper) end of
                the confidence region.

        """
        std2 = self.stddev.mul_(2)
        mean = self.mean
        return mean.sub(std2), mean.add(std2)

    @property
    def islazy(self):
        return self._islazy

    def __add__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()
