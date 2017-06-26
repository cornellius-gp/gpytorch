import torch
from torch.autograd import Variable
from torch.nn import Parameter
from gpytorch.math.functions import AddDiag
from gpytorch.distributions import Distribution
from gpytorch.random_variables import GaussianRandomVariable


class GaussianLikelihood(Distribution):
    def __init__(self):
        super(GaussianLikelihood, self).__init__()
        self.log_noise = Parameter(torch.zeros(1, 1))
        self.initialize()


    def initialize(self, log_noise=0):
        self.log_noise.data.fill_(log_noise)
        return self


    def forward(self, input):
        assert(isinstance(input, GaussianRandomVariable))
        mean, covar = input.representation()
        noise = AddDiag()(covar, self.log_noise.exp())
        return GaussianRandomVariable(mean, covar + noise)

