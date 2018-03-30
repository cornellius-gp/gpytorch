from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .marginal_log_likelihood import MarginalLogLikelihood


class VariationalMarginalLogLikelihood(MarginalLogLikelihood):
    def __init__(self, likelihood, model, n_data):
        """
        A special MLL designed for variational inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - n_data: (int) - the total number of training data points (necessary for SGD)
        """
        super(VariationalMarginalLogLikelihood, self).__init__(likelihood, model)
        self.n_data = n_data

    def forward(self, output, target, **kwargs):
        n_batch = target.size(0)

        log_likelihood = self.likelihood.log_probability(output, target, **kwargs).div(n_batch)
        kl_divergence = sum(variational_strategy.kl_divergence().sum()
                            for variational_strategy in self.model.variational_strategies()).div(self.n_data)

        res = log_likelihood - kl_divergence
        return res
