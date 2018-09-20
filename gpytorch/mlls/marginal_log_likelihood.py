from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ..module import Module


class MarginalLogLikelihood(Module):
    """
    A module to compute marginal log likelihoods of data

    Args:
    - likelihood: (Likelihood) - the likelihood for the model
    - model: (Module) - the GP model
    """

    def __init__(self, likelihood, model):
        super(MarginalLogLikelihood, self).__init__()
        self.likelihood = likelihood
        self.model = model

    def forward(self, output, target):
        """
        Args:
        - output: (MultivariateNormal) - the outputs of the latent function
        - target: (Variable) - the target values
        """
        raise NotImplementedError
