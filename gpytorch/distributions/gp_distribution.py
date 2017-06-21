import torch
from torch.nn import Parameter
from torch.autograd import Variable
from .distribution import Distribution
from gpytorch.math.functions import Invmm
from gpytorch.random_variables import RandomVariable, GaussianRandomVariable


class GPDistribution(Distribution):
    def __init__(self, mean_module=None, covar_module=None):
        super(GPDistribution, self).__init__()
        self.mean_module = mean_module
        self.covar_module = covar_module

        # Buffers for conditioning on data
        self.register_buffer('train_x', torch.Tensor())
        self.register_buffer('train_covar', torch.Tensor())
        self.register_buffer('alpha', torch.Tensor())

        # Wrap data-conditioned buffers in variables
        self.train_x_var = Variable(self.train_x)
        self.train_covar_var = Variable(self.train_covar)
        self.alpha_var = Variable(self.alpha)


    def forward(self, input):
        assert(isinstance(input, Variable))
        test_test_covar = self.forward_covar(input, input)
        test_mean = self.forward_mean(input)

        # Utilize training data, if conditioned on training data
        if len(self.alpha) and len(self.train_x) and len(self.train_covar):
            train_test_covar = self.forward_covar(input, self.train_x_var)
            test_mean = test_mean.add(
                torch.mv(train_test_covar, self.alpha_var)
            )
            test_test_covar = test_test_covar.sub(
                torch.mm(train_test_covar, Invmm()(self.train_covar_var, train_test_covar.t()))
            )
            
        return GaussianRandomVariable(test_mean, test_test_covar)


    def forward_covar(self, input_1, input_2):
        return self.covar_module(input_1, input_2)


    def forward_mean(self, input):
        return self.mean_module(input)
