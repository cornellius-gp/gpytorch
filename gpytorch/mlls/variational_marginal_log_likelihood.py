from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .marginal_log_likelihood import MarginalLogLikelihood


class VariationalMarginalLogLikelihood(MarginalLogLikelihood):
    def __init__(self, likelihood, model, num_data, combine_terms=True):
        """
        A special MLL designed for variational inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - num_data: (int) - the total number of training data points (necessary for SGD)
        - combine_terms: (bool) - whether or not to sum the expected NLL with the KL terms (default True)
        """
        super(VariationalMarginalLogLikelihood, self).__init__(likelihood, model)
        self.num_data = num_data
        self.combine_terms = combine_terms

    def forward(self, output, target, **kwargs):
        num_batch = target.size(0)

        log_likelihood = self.likelihood.variational_log_probability(output, target, **kwargs).div(num_batch)
        kl_divergence = sum(
            variational_strategy.kl_divergence().sum() for variational_strategy in self.model.variational_strategies()
        ).div(self.num_data)

        if self.combine_terms:
            res = log_likelihood - kl_divergence
            for _, param, prior in self.named_parameter_priors():
                res.add_(prior.log_prob(param).sum())
            for _, params, transform, prior in self.named_derived_priors():
                res.add_(prior.log_prob(transform(*params)).sum())
            return res
        else:
            log_prior = torch.zeros_like(log_likelihood)
            for _, param, prior in self.named_parameter_priors():
                log_prior.add_(prior.log_prob(param).sum())
            for _, params, transform, prior in self.named_derived_priors():
                log_prior.add_(prior.log_prob(transform(*params)).sum())
            return log_likelihood, kl_divergence, log_prior
