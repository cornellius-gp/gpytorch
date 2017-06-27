import math
import torch
from torch.nn import Module
from gpytorch.math.functions import ExactGPMarginalLogLikelihood as LogLikelihoodFunc

class ExactGPMarginalLogLikelihood(Module):
    def forward(self, covar, y):
        return LogLikelihoodFunc()(covar, y)
