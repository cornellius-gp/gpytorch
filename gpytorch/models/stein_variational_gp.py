#!/usr/bin/env python3

import torch
import pyro
from .. import Module
from ..lazy import RootLazyTensor, DiagLazyTensor
from ..distributions import MultivariateNormal


class SteinVariationalGP(Module):
    def __init__(self, inducing_points, likelihood, num_data, name_prefix=""):
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        super().__init__()
        self.likelihood = likelihood
        self.num_data = num_data
        self.name_prefix = name_prefix

        # Cheap buffers
        self.register_parameter("inducing_points", torch.nn.Parameter(inducing_points))
        self.register_buffer("prior_mean", torch.zeros(inducing_points.shape[:-1]))
        self.register_buffer("prior_var", torch.ones(inducing_points.shape[:-1]))

    def model(self, input, output, *params, **kwargs):
        pyro.module(self.name_prefix + ".gp_prior", self)

        function_dist = self(input, *params, **kwargs)
        function_dist = pyro.distributions.Normal(function_dist.mean, function_dist.variance.sqrt())

        # Go from function -> output
        num_minibatch = function_dist.batch_shape[-1]
        with pyro.poutine.scale(scale=float(self.num_data / num_minibatch)):
            return self.likelihood.pyro_sample_output(
                output, function_dist, *params, **kwargs
            )

    def sample_inducing_values(self):
        """
        Sample values from the inducing point distribution `p(u)` or `q(u)`.
        This should only be re-defined to note any conditional independences in
        the `inducing_values_dist` distribution. (By default, all batch dimensions
        are not marked as conditionally indendent.)
        """
        prior_dist = MultivariateNormal(self.prior_mean, DiagLazyTensor(self.prior_var))
        samples = pyro.sample(self.name_prefix + ".inducing_values", prior_dist)
        return samples

    def __call__(self, input, *args, **kwargs):
        # Draw samples from p(u) for KL divergence computation
        inducing_values_samples = self.sample_inducing_values()

        # Get function dist
        num_induc = self.inducing_points.size(-2)
        full_inputs = torch.cat([self.inducing_points, input], dim=-2)
        full_output = self.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        test_mean = full_output.mean[..., num_induc:]
        L = full_covar[..., :num_induc, :num_induc].add_jitter().cholesky().evaluate()
        Linv = torch.triangular_solve(torch.eye(L.size(-1), dtype=L.dtype, device=L.device), L, upper=False)[0]
        cross_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        scaled_cross_covar = Linv @ cross_covar
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        function_dist = MultivariateNormal(
            torch.squeeze(scaled_cross_covar.transpose(-1, -2) @ inducing_values_samples.unsqueeze(-1)),
            data_data_covar + RootLazyTensor(scaled_cross_covar.transpose(-1, -2)).mul(-1)
        )
        return function_dist
