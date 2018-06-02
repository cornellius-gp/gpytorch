from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .sum_lazy_variable import SumLazyVariable


class PsdSumLazyVariable(SumLazyVariable):
    """
    A SumLazyVariable, but where every component of the sum is positive semi-definite
    """

    def zero_mean_mvn_samples(self, n_samples):
        return sum(lazy_var.zero_mean_mvn_samples(n_samples) for lazy_var in self.lazy_vars)
