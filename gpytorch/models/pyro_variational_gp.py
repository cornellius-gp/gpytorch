from __future__ import absolute_import, division, print_function, unicode_literals

from .abstract_variational_gp import AbstractVariationalGP
import pyro


class PyroVariationalGP(AbstractVariationalGP):
    def __init__(self, variational_strategy, likelihood, name_prefix=""):
        super(PyroVariationalGP, self).__init__(variational_strategy)
        self.name_prefix = name_prefix
        self.likelihood = likelihood

    def guide(self, x, y):
        variational_dist = self.variational_strategy.variational_distribution.variational_distribution
        pyro.sample(self.name_prefix + "._inducing_values", variational_dist)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp_prior", self)
        variational_dist_f = self(x)
        prior_dist = self.variational_strategy.prior_distribution
        inducing_value_samples = pyro.sample(self.name_prefix + "._inducing_values", prior_dist)
        sample_shape = inducing_value_samples.shape[: inducing_value_samples.dim() - len(prior_dist.shape())]
        self.likelihood.pyro_sample_y(variational_dist_f, y, sample_shape, self.name_prefix)
