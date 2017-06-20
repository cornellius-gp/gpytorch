import torch
from torch.nn import Parameter
from torch.autograd import Variable
from .distribution import Distribution
from gpytorch.random_variables import RandomVariable, GaussianRandomVariable


class GPDistribution(Distribution):
    def __init__(self, mean_module=None, covar_module=None):
        super(GPDistribution, self).__init__()
        self.mean_module = mean_module
        self.covar_module = covar_module


    def forward(self, input):
        assert(isinstance(input, Variable))
        test_test_covar = self.forward_covar(input, input)
        test_mean = self.forward_mean(input)
            
        return GaussianRandomVariable(test_mean, test_test_covar)


    def forward_covar(self, input_1, input_2):
        return self.covar_module(input_1, input_2)


    def forward_mean(self, input):
        return self.mean_module(input)
