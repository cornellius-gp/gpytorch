#!/usr/bin/env python3

import pyro
import torch


class _PyroMixin(object):
    def pyro_factors(self, beta=1.0, name_prefix=""):
        # Include factor for KL[ q(u) || p(u) ]
        # We compute this analytically
        kl = self.variational_strategy.kl_divergence()
        with pyro.poutine.scale(scale=beta):
            pyro.factor(name_prefix + ".kl_divergence_u", -kl)

        # Include term for GPyTorch priors
        log_prior = torch.tensor(0., dtype=kl.dtype, device=kl.device)
        for _, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure()).sum().div(self.num_data))
        pyro.factor(name_prefix + ".log_prior", log_prior)

        # Include factor for added loss terms
        added_loss = torch.tensor(0., dtype=kl.dtype, device=kl.device)
        for added_loss_term in self.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
        pyro.factor(name_prefix + ".added_loss", added_loss)

    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        # HACK - this is a way to get the sample shape
        pyro.sample("__throwaway__", pyro.distributions.Normal(0, 1))

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        # Include KL[ q(u) || p(u) ], GPyTorch priors, etc.
        self.pyro_factors(beta=beta, name_prefix=name_prefix)

        # HACK - this is a way to get the sample shape
        sample_shape = pyro.sample("__throwaway__", pyro.distributions.Normal(0, 1)).shape

        # Draw samples from q(f)
        function_dist = self(input)
        function_dist = pyro.distributions.Normal(
            loc=function_dist.mean,
            scale=function_dist.stddev,
        ).to_event(len(function_dist.event_shape) - 1)
        return function_dist(sample_shape[:-len(function_dist.batch_shape)])
