#!/usr/bin/env python3

from ..variational import CholeskyVariationalDistribution, VariationalStrategy
from .abstract_variational_gp import AbstractVariationalGP
import warnings


class VariationalGP(AbstractVariationalGP):
    def __init__(self, train_input):
        warnings.warn(
            "VariationalGP is deprecated in favor of a new variational inference interface, and will "
            "be removed in a future release. Please see the new examples.",
            DeprecationWarning,
        )
        variational_distribution = CholeskyVariationalDistribution(train_input.size(0))
        variational_strategy = VariationalStrategy(self, train_input, variational_distribution)
        super(VariationalGP, self).__init__(variational_strategy)
