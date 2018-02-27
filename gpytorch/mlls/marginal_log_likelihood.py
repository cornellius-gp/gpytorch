from .. import Module


class MarginalLogLikelihood(Module):
    def __init__(self, likelihood, model):
        """
        A module to compute marginal log likelihoods of data

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the GP model
        """

        super(MarginalLogLikelihood, self).__init__()
        self.likelihood = likelihood
        self.model = model

    def forward(self, output, target):
        """
        Args:
        - output: (GaussianRandomVariable) - the outputs of the latent function
        - target: (Variable) - the target values
        """
        raise NotImplementedError
