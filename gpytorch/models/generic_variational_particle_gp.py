#!/usr/bin/env python3
import math
import torch
import numpy as np
import pyro
from .. import Module
from ..lazy import RootLazyTensor, DiagLazyTensor, BlockDiagLazyTensor
from ..distributions import MultivariateNormal, MultitaskMultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from . import GP
from .. import settings
from .quad_prob import quad_prob

from numpy.polynomial.hermite import hermgauss


class GenericVariationalParticleGP(Module):
    def __init__(self, inducing_points, likelihood, num_data, name_prefix="",
                 mode="jensen", beta=1.0, divbeta=0.05):
        assert mode in ['pred_class', 'jensen', 'predictive', 'tpredictive',
                        'betadiv', 'gammadiv', 'robust', 'class_gamma', 'class_svi']

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

        if self.mode == 'tpredictive':
            self.raw_nu = torch.nn.Parameter(torch.tensor(1.0))

        if self.mode == 'pred_class':
            NQ = 15
            quad_X, quad_WX = hermgauss(NQ)
            self.register_buffer("quad_X", torch.tensor(quad_X))
            self.register_buffer("quad_WX", torch.tensor(quad_WX))

    def model(self, input, output, *params, **kwargs):
        pyro.module(self.name_prefix + ".gp_prior", self)

        function_dist = self(input, *params, **kwargs)

        # Go from function -> output
        num_minibatch = function_dist.event_shape[0]
        scale_factor = float(1.0 / num_minibatch)

        if self.mode == 'predictive':
            obs_dist = pyro.distributions.Normal(function_dist.mean,
                (function_dist.variance + self.likelihood.noise).sqrt()).to_event(function_dist.mean.dim())
            with pyro.poutine.scale(scale=scale_factor):
                return pyro.sample(self.name_prefix + ".output_values", obs_dist, obs=output)
        elif self.mode == 'class_svi':
            with pyro.poutine.scale(scale=scale_factor), settings.fast_computations(covar_root_decomposition=False):
                function_samples = function_dist()
                output_dist = self.likelihood(function_samples, *params, **kwargs)
                pyro.sample(self.name_prefix + ".output_values", output_dist, obs=output)
        elif self.mode == 'pred_class':
            muf, varf = function_dist.mean.t(), function_dist.variance.t()
            log_prob = quad_prob(muf, varf, output, K=3, X=self.quad_X, WX=self.quad_WX).clamp(min=1.0e-8).log()
            pyro.factor(self.name_prefix + ".output_values", scale_factor * log_prob)
        elif self.mode == 'class_gamma':
            with settings.fast_computations(covar_root_decomposition=False):
                function_samples = function_dist()
            probs = self.likelihood(function_samples, *params, **kwargs).probs
            gamma = self.divbeta
            probs_log = probs.log()
            log_num = gamma * probs_log[torch.arange(output.size(0)), output]
            log_den = (gamma / (1.0 + gamma)) * torch.logsumexp((1.0 + gamma) * probs_log, -1)
            factor = scale_factor * ((1.0 + gamma) / gamma) * (log_num - log_den).exp()
            pyro.factor(self.name_prefix + ".output_values", factor)
        elif self.mode == 'tpredictive':
            nu = 2.0 + torch.functional.F.softplus(pyro.param("raw_nu", self.raw_nu))
            obs_dist = pyro.distributions.StudentT(nu, function_dist.mean,
                (function_dist.variance + self.likelihood.noise).sqrt()).to_event(1)
            with pyro.poutine.scale(scale=scale_factor):
                return pyro.sample(self.name_prefix + ".output_values", obs_dist, obs=output)
        elif self.mode == 'jensen':
            obs_dist = torch.distributions.Normal(function_dist.mean, self.likelihood.noise.sqrt())
            factor1 = obs_dist.log_prob(output).sum(-1)
            factor2 = 0.5 * function_dist.variance.sum(-1) / self.likelihood.noise
            factor = scale_factor * (factor1 - factor2)
            pyro.factor(self.name_prefix + ".output_values", factor)
        elif self.mode == 'robust':
            # adapted from https://github.com/JeremiasKnoblauch/GVIPublic/
            gamma, noise = self.divbeta, self.likelihood.noise
            muf, varf = function_dist.mean, function_dist.variance
            mut = gamma * output / noise + muf / varf
            sigmat = 1.0 / (gamma / noise + 1.0 / varf)

            log_integral = -0.5 * gamma * torch.log(2.0 * math.pi * noise) - 0.5 * np.log1p(gamma)
            log_tempered = -math.log(gamma) \
                           - 0.5 * gamma * torch.log(2.0 * math.pi * noise) \
                           - 0.5 * torch.log1p(gamma * varf / noise) \
                           - 0.5 * (gamma * output.pow(2.0) / noise) \
                           - 0.5 * muf.pow(2.0) / varf \
                           + 0.5 * mut.pow(2.0) * sigmat

            factor = log_tempered + gamma / (1.0 + gamma) * log_integral + (1.0 + gamma)
            if muf.dim() == 2:
                pyro.factor(self.name_prefix + ".output_values", scale_factor * factor.sum(0).exp().sum(-1))
            else:
                pyro.factor(self.name_prefix + ".output_values", scale_factor * factor.exp().sum(-1))
        elif self.mode == 'betadiv':
            pred_variance = function_dist.variance + self.likelihood.noise
            obs_dist = torch.distributions.Normal(function_dist.mean, pred_variance.sqrt())
            term1 = (self.divbeta * obs_dist.log_prob(output)).exp().sum(-1) / self.divbeta
            term2num = -math.pow(math.sqrt(1.0 + self.divbeta), self.divbeta - 1.0)
            term2den = torch.pow(2.0 * math.pi * pred_variance, 0.5 * self.divbeta)
            term2 = (term2num / term2den).sum(-1)
            factor = scale_factor * (term1 + term2)
            pyro.factor(self.name_prefix + ".output_values", factor)
        elif self.mode == 'gammadiv':
            pred_variance = function_dist.variance + self.likelihood.noise
            obs_dist = torch.distributions.Normal(function_dist.mean, pred_variance.sqrt())
            term1 = obs_dist.log_prob(output).sum(-1)
            term2 = (0.5 * math.pow(self.divbeta, 3.0) / (1.0 + self.divbeta)) * pred_variance.log().sum(-1)
            factor = scale_factor * (term1 + term2)
            pyro.factor(self.name_prefix + ".output_values", factor)

    def sample_inducing_values(self):
        """
        Sample values from the inducing point distribution `p(u)` or `q(u)`.
        This should only be re-defined to note any conditional independences in
        the `inducing_values_dist` distribution. (By default, all batch dimensions
        are not marked as conditionally indendent.)
        """
        beta = self.beta if self.beta > 0.0 else 1.0e-20
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

        # in practice this will return MAP/SVGD particle(s)
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
