import torch
from torch.autograd import Variable
from torch.nn import Parameter
from gpytorch.math.functions import Diag
from gpytorch.distributions import Distribution
from gpytorch.random_variables import GaussianRandomVariable


class GaussianLikelihood(Distribution):
    def __init__(self):
        super(GaussianLikelihood, self).__init__()
        self.noise = Parameter(torch.zeros(1, 1))
        self.initialize()


    def initialize(self, noise=0):
        self.noise.data.fill_(noise)
        return self


    def forward(self, input):
        assert(isinstance(input, GaussianRandomVariable))
        mean, covar = input.representation()
        noise = Diag(covar.size(0))(self.noise)
        return GaussianRandomVariable(mean, covar + noise)

