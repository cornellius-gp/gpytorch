from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .marginal_log_likelihood import MarginalLogLikelihood
from ..likelihoods import GaussianLikelihood
from ..distributions import MultivariateNormal
from ..variational import MVNVariationalStrategy


class ExactMarginalLogLikelihood(MarginalLogLikelihood):
    def __init__(self, likelihood, model):
        """
        A special MLL designed for exact inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the exact GP model
        """
        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, output, target, *params):
        if not isinstance(output, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(output, *params)
        res = output.log_prob(target)

        # Add terms for SGPR / when inducing points are learned
        trace_diff = torch.zeros_like(res)
        for variational_strategy in self.model.variational_strategies():
            if isinstance(variational_strategy, MVNVariationalStrategy):
                trace_diff = trace_diff.add(variational_strategy.trace_diff())
        if hasattr(self.likelihood, "log_noise"):
            trace_diff = trace_diff / self.likelihood.log_noise.exp()
            res = res.add(0.5, trace_diff)

        # Add log probs of priors on the parameters
        for _, param, prior in self.named_parameter_priors():
            res.add_(prior.log_prob(param).sum())
        for _, prior, params, transform in self.named_derived_priors():
            res.add_(prior.log_prob(transform(*params)).sum())

        # Scale by the amount of data we have
        num_data = target.size(-1)
        return res.div_(num_data)
