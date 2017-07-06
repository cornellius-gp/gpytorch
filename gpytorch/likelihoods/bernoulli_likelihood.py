import torch
from torch.autograd import Variable
from torch.nn import Parameter
from gpytorch.math.functions import AddDiag, ExactGPMarginalLogLikelihood, NormalCDF, LogNormalCDF
from gpytorch.random_variables import GaussianRandomVariable, IndependentRandomVariables, BernoulliRandomVariable
from .likelihood import Likelihood


class BernoulliLikelihood(Likelihood):
    def forward(self, input):
        if not isinstance(input, GaussianRandomVariable):
            raise RuntimeError('BernoulliLikelihood expects a Gaussian distributed latent function to make predictions')

        mean = input.mean()
        var = input.var()

        link = mean.div(torch.sqrt(1+var))

        output_probs = NormalCDF()(link)
        return IndependentRandomVariables([BernoulliRandomVariable(output_prob) for output_prob in output_probs])

    def log_probability(self, f, y):
    	return LogNormalCDF()(f.mul(y.unsqueeze(1).expand_as(f))).mean(1).sum(0)
