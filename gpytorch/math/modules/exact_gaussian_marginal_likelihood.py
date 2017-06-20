import math
import torch
from torch.nn import Module
from gpytorch.math.functions import LogDet, InverseQuadForm

class MarginalLikelihood(Module):
    def forward(self, covar, y):
        res = InverseQuadForm(y)(covar) + LogDet()(covar)
        res.add_(math.log(2 * math.pi) * len(y))
        res.mul_(-0.5)
        return res
