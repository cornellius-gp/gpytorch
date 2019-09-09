#!/usr/bin/env python3
import math
import torch
import pyro
from .. import Module
from ..lazy import RootLazyTensor, DiagLazyTensor, BlockDiagLazyTensor
from ..distributions import MultivariateNormal, MultitaskMultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from . import GP
from .. import settings


class SteinVariationalGP(Module):
    def __init__(self, inducing_points, likelihood, num_data, name_prefix="",
                 mode="jensen", beta=1.0, divbeta=0.1):
        assert mode in ['jensen', 'predictive', 'betadiv']

        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        self.mode = mode
        self.beta = beta
        self.divbeta = divbeta

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

        # Go from function -> output
        num_minibatch = function_dist.event_shape[0]
        scale_factor = float(1.0 / num_minibatch)

        if self.mode == 'predictive':
            obs_dist = pyro.distributions.Normal(function_dist.mean,
                (function_dist.variance + self.likelihood.noise).sqrt()).to_event(1)
            with pyro.poutine.scale(scale=scale_factor):
                return pyro.sample(self.name_prefix + ".output_values", obs_dist, obs=output)
        elif self.mode == 'jensen':
            obs_dist = torch.distributions.Normal(function_dist.mean, self.likelihood.noise.sqrt())
            factor1 = obs_dist.log_prob(output).sum(-1)
            factor2 = 0.5 * function_dist.variance.sum(-1) / self.likelihood.noise
            factor = scale_factor * (factor1 - factor2)
            pyro.factor(self.name_prefix + ".output_values", factor)
        elif self.mode == 'betadiv':
            obs_dist = torch.distributions.Normal(function_dist.mean,
                (function_dist.variance + self.likelihood.noise).sqrt())
            term1 = (self.divbeta * obs_dist.log_prob(output)).exp().sum(-1) / self.divbeta
            term2num = -math.pow(math.sqrt(1.0 + self.divbeta), self.divbeta - 1.0)
            term2den = torch.pow(2.0 * math.pi * function_dist.variance, 0.5 * self.divbeta)
            term2 = (term2num / term2den).sum(-1)
            factor = scale_factor * (term1 + term2)
            pyro.factor(self.name_prefix + ".output_values", factor)

    def sample_inducing_values(self):
        """
        Sample values from the inducing point distribution `p(u)` or `q(u)`.
        This should only be re-defined to note any conditional independences in
        the `inducing_values_dist` distribution. (By default, all batch dimensions
        are not marked as conditionally indendent.)
        """
        beta = self.beta if self.beta > 0.0 else 1.0e-10
        prior_dist = MultivariateNormal(self.prior_mean, DiagLazyTensor(self.prior_var))
        with pyro.poutine.scale(scale=beta / self.num_data):
            return pyro.sample(self.name_prefix + ".inducing_values", prior_dist)

    def __call__(self, input, *args, **kwargs):
        inducing_points = self.inducing_points
        inducing_batch_shape = inducing_points.shape[:-2]
        if inducing_batch_shape < input.shape[:-2]:
            batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], input.shape[:-2])
            inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
            input = input.expand(*batch_shape, *input.shape[-2:])
        # Draw samples from p(u) for KL divergence computation
        inducing_values_samples = self.sample_inducing_values()

        # Get function dist
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, input], dim=-2)
        full_output = self.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        test_mean = full_output.mean[..., num_induc:]
        L = full_covar[..., :num_induc, :num_induc].add_jitter().cholesky().evaluate()
        cross_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        scaled_cross_covar = torch.triangular_solve(cross_covar, L, upper=False)[0]
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        function_dist = MultivariateNormal(
            (scaled_cross_covar.transpose(-1, -2) @ inducing_values_samples.unsqueeze(-1)).squeeze(-1),
            data_data_covar + RootLazyTensor(scaled_cross_covar.transpose(-1, -2)).mul(-1)
        )
        return function_dist
