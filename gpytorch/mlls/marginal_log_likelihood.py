#!/usr/bin/env python3

from ..module import Module
from ..models import GP


class MarginalLogLikelihood(Module):
    """
    A module to compute marginal log likelihoods of data

    Args:
    - likelihood: (Likelihood) - the likelihood for the model
    - model: (Module) - the GP model
    """

    def __init__(self, likelihood, model):
        super(MarginalLogLikelihood, self).__init__()
        if not isinstance(model, GP):
            raise RuntimeError(
                "All MarginalLogLikelihood objects must be given a GP object as a model. If you are "
                "using a more complicated model involving a GP, pass the underlying GP object as the "
                "model, not a full PyTorch module."
            )
        self.likelihood = likelihood
        self.model = model

    def forward(self, output, target):
        """
        Args:
        - output: (MultivariateNormal) - the outputs of the latent function
        - target: (Variable) - the target values
        """
        raise NotImplementedError
