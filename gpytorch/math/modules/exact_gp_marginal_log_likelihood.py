import math
import torch
from torch.nn import Module
from gpytorch.math.functions import LogDet, InverseQuadForm

class ExactGPMarginalLogLikelihood(Module):
    def forward(self, covar, y):
        res = InverseQuadForm()(covar, y) + LogDet()(covar)
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5
        return res
