import math
import torch
from torch.nn import Module
from gpytorch.math.functions import LogDet, Invmv

class ExactGPMarginalLogLikelihood(Module):
    def forward(self, covar, y):
        res = Invmv()(covar, y).dot(y) + LogDet()(covar)
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5
        return res
