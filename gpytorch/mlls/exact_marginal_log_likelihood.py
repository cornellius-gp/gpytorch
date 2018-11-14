from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .marginal_log_likelihood import MarginalLogLikelihood
from ..likelihoods import GaussianLikelihood
from ..distributions import MultivariateNormal


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
        if not isinstance(output, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(output)
        res = output.log_prob(target)

        # Add terms for SGPR / when inducing points are learned
        added_loss = torch.zeros_like(res)
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add(added_loss_term.loss())

        res = res.add(0.5, added_loss)

        # Add log probs of priors on the parameters
        for _, prior, params_and_tfs, prior_tf in self.named_priors():
            params = [p if ptf is None else ptf(p) for p, ptf in params_and_tfs]
            lp_arg = params[0] if prior_tf is None else prior_tf(*params)
            res.add_(prior.log_prob(lp_arg).sum())

        # Scale by the amount of data we have
        num_data = target.size(-1)
        return res.div_(num_data)
