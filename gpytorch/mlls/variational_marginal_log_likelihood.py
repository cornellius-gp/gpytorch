#!/usr/bin/env python3

from .variational_elbo import VariationalELBO


class VariationalMarginalLogLikelihood(VariationalELBO):
    def __init__(self, likelihood, model, num_data, combine_terms=True):
        """
        A special MLL designed for variational inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - num_data: (int) - the total number of training data points (necessary for SGD)
        - combine_terms: (bool) - whether or not to sum the expected NLL with the KL terms (default True)
        """
        super(VariationalMarginalLogLikelihood, self).__init__(likelihood, model, num_data, combine_terms)
