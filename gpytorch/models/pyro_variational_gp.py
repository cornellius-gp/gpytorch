#!/usr/bin/env python3

import pyro
from .abstract_variational_gp import AbstractVariationalGP


class PyroVariationalGP(AbstractVariationalGP):
    def __init__(self, variational_strategy, likelihood, num_data, name_prefix=""):
        super(PyroVariationalGP, self).__init__(variational_strategy)
        self.name_prefix = name_prefix
        self.likelihood = likelihood
        self.num_data = num_data

    def sample_inducing_values(self, inducing_values_dist):
        reinterpreted_batch_ndims = len(inducing_values_dist.batch_shape)
        samples = pyro.sample(
            self.name_prefix + ".inducing_values",
            inducing_values_dist.to_event(reinterpreted_batch_ndims)
        )
        return samples

    def guide(self, x, y):
        variational_dist_u = self.variational_strategy.variational_distribution.variational_distribution
        # Draw samples from q(u) for KL divergence computation
        self.sample_inducing_values(variational_dist_u)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp_prior", self)
        prior_dist = self.variational_strategy.prior_distribution

        # Draw samples from p(u) for KL divergence computation
        self.sample_inducing_values(prior_dist)

        # Get the variational distribution for the function
        variational_dist_f = self(x)
        num_minibatch = variational_dist_f.event_shape.numel()

        # Now make the variational distribution Normal - for conditional indepdence
        variational_dist_f = pyro.distributions.Normal(variational_dist_f.mean, variational_dist_f.variance)

        with pyro.poutine.scale(scale=float(self.num_data / num_minibatch)):
            return self.likelihood.pyro_sample_outputs(y, variational_dist_f)
