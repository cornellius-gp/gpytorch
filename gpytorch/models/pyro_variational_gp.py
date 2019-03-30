#!/usr/bin/env python3

import torch
import pyro
from .abstract_variational_gp import AbstractVariationalGP


class PyroVariationalGP(AbstractVariationalGP):
    def __init__(self, variational_strategy, likelihood, num_data, name_prefix=""):
        super(PyroVariationalGP, self).__init__(variational_strategy)
        self.name_prefix = name_prefix
        self.likelihood = likelihood
        self.num_data = num_data

    def guide(self, input, output, *params, **kwargs):
        inducing_dist = self.variational_strategy.variational_distribution.variational_distribution
        # Draw samples from q(u) for KL divergence computation
        self.sample_inducing_values(inducing_dist)
        self.likelihood.guide(*params, **kwargs)

    def model(self, input, output, *params, **kwargs):
        pyro.module(self.name_prefix + ".gp_prior", self)
        prior_dist = self.variational_strategy.prior_distribution

        # Draw samples from p(u) for KL divergence computation
        inducing_values_samples = self.sample_inducing_values(prior_dist)
        sample_shape = inducing_values_samples.shape[:-len(prior_dist.shape())] + \
            torch.Size([1] * len(prior_dist.batch_shape))

        # Get the variational distribution for the function
        function_dist = self(input)

        # Go from function -> output
        num_minibatch = function_dist.batch_shape[-1]
        with pyro.poutine.scale(scale=float(self.num_data / num_minibatch)):
            return self.likelihood.pyro_sample_output(
                output, function_dist, *params, **kwargs, sample_shape=sample_shape
            )

    def sample_inducing_values(self, inducing_values_dist):
        """
        Sample values from the inducing point distribution `p(u)` or `q(u)`.
        This should only be re-defined to note any conditional independences in
        the `inducing_values_dist` distribution. (By default, all batch dimensions
        are not marked as conditionally indendent.)
        """
        reinterpreted_batch_ndims = len(inducing_values_dist.batch_shape)
        samples = pyro.sample(
            self.name_prefix + ".inducing_values",
            inducing_values_dist.to_event(reinterpreted_batch_ndims)
        )
        return samples

    def transform_function_dist(self, function_dist):
        """
        Transform the function_dist from `gpytorch.distributions.MultivariateNormal` into
        some other variant of a Normal distribution.

        This is useful for marking conditional independencies (useful for inference),
        marking that the distribution contains multiple outputs, etc.

        By default, this funciton transforms a multivariate normal into a set of conditionally
        independent normals when performing inference, and keeps the distribution multivariate
        normal for predictions.
        """
        if self.training:
            return pyro.distributions.Normal(function_dist.mean, function_dist.variance)
        else:
            return function_dist

    def __call__(self, input, *args, **kwargs):
        function_dist = super().__call__(input, *args, **kwargs)
        # Now make the variational distribution Normal - for conditional indepdence
        function_dist = self.transform_function_dist(function_dist)
        res = function_dist
        return res
