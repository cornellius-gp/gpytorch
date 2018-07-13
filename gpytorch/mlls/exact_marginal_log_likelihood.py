from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from .marginal_log_likelihood import MarginalLogLikelihood
from ..lazy import LazyVariable, NonLazyVariable
from ..likelihoods import GaussianLikelihood
from ..random_variables import GaussianRandomVariable
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

    def forward(self, output, target):
        if not isinstance(output.covar(), LazyVariable):
            output = GaussianRandomVariable(output.mean(), NonLazyVariable(output.covar()))
        mean, covar = self.likelihood(output).representation()
        n_data = target.size(-1)

        # Get log determininat and first part of quadratic form
        inv_quad, log_det = covar.inv_quad_log_det(inv_quad_rhs=(target - mean).unsqueeze(-1), log_det=True)

        # Add terms for SGPR / when inducing points are learned
        trace_diff = torch.zeros_like(inv_quad)
        for variational_strategy in self.model.variational_strategies():
            if isinstance(variational_strategy, MVNVariationalStrategy):
                trace_diff = trace_diff.add(variational_strategy.trace_diff())
        trace_diff = trace_diff / self.likelihood.log_noise.exp()

        res = -0.5 * sum([inv_quad, log_det, n_data * math.log(2 * math.pi), -trace_diff])
        res.div_(n_data)

        # Add log probs of priors on the parameters
        for _, param, prior in self.named_parameter_priors():
            res.add_(prior.log_prob(param).sum())
        for _, params, transform, prior in self.named_derived_priors():
            res.add_(prior.log_prob(transform(*params)).sum())

        return res
