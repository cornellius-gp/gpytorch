#!/usr/bin/env python3

import torch

import pyro


class _PyroMixin(object):
    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        # Inducing values q(u)
        with pyro.poutine.scale(scale=beta):
            variational_distribution = self.variational_strategy.variational_distribution
            variational_distribution = variational_distribution.to_event(len(variational_distribution.batch_shape))
            pyro.sample(name_prefix + ".u", variational_distribution)

        # Draw samples from q(f)
        function_dist = self(input, prior=False)
        function_dist = pyro.distributions.Normal(loc=function_dist.mean, scale=function_dist.stddev).to_event(
            len(function_dist.event_shape) - 1
        )
        return function_dist.mask(False)

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        # Inducing values p(u)
        with pyro.poutine.scale(scale=beta):
            prior_distribution = self.variational_strategy.prior_distribution
            prior_distribution = prior_distribution.to_event(len(prior_distribution.batch_shape))
            u_samples = pyro.sample(name_prefix + ".u", prior_distribution)

        # Include term for GPyTorch priors
        log_prior = torch.tensor(0.0, dtype=u_samples.dtype, device=u_samples.device)
        for _, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum())
        pyro.factor(name_prefix + ".log_prior", log_prior)

        # Include factor for added loss terms
        added_loss = torch.tensor(0.0, dtype=u_samples.dtype, device=u_samples.device)
        for added_loss_term in self.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
        pyro.factor(name_prefix + ".added_loss", added_loss)

        # Draw samples from p(f)
        function_dist = self(input, prior=True)
        function_dist = pyro.distributions.Normal(loc=function_dist.mean, scale=function_dist.stddev).to_event(
            len(function_dist.event_shape) - 1
        )
        return function_dist.mask(False)
