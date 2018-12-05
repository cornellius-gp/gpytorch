#!/usr/bin/env python3

from .abstract_variational_gp import AbstractVariationalGP
from ..lazy import BlockDiagLazyTensor
import pyro


class PyroVariationalGP(AbstractVariationalGP):
    def __init__(self, variational_strategy, likelihood, num_data, name_prefix=""):
        super(PyroVariationalGP, self).__init__(variational_strategy)
        self.name_prefix = name_prefix
        self.likelihood = likelihood
        self.num_data = num_data

    def guide(self, x, y):
        variational_dist = self.variational_strategy.variational_distribution.variational_distribution
        if len(variational_dist.batch_shape):
            variational_dist = variational_dist.__class__(
                variational_dist.mean.contiguous().view(-1),
                BlockDiagLazyTensor(variational_dist.lazy_covariance_matrix),
            )
        pyro.sample(self.name_prefix + "._inducing_values", variational_dist)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp_prior", self)
        variational_dist_f = self(x)
        prior_dist = self.variational_strategy.prior_distribution
        if len(prior_dist.batch_shape):
            prior_dist = prior_dist.__class__(
                prior_dist.mean.contiguous().view(-1), BlockDiagLazyTensor(prior_dist.lazy_covariance_matrix)
            )
        inducing_value_samples = pyro.sample(self.name_prefix + "._inducing_values", prior_dist)
        sample_shape = inducing_value_samples.shape[: inducing_value_samples.dim() - len(prior_dist.shape())]

        num_minibatch = variational_dist_f.event_shape.numel()
        with pyro.poutine.scale(scale=float(self.num_data / num_minibatch)):
            self.likelihood.pyro_sample_y(variational_dist_f, y, sample_shape, self.name_prefix)
