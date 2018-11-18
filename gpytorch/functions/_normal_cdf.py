#!/usr/bin/env python3

import torch
from torch.autograd import Function
import math


class NormalCDF(Function):
    def __init__(self):
        self.a_for_erf = 8.0 / (3.0 * math.pi) * (math.pi - 3.0) / (4.0 - math.pi)

    def erf_approx(self, x):
        exp = -x * x * (4 / math.pi + self.a_for_erf * x * x) / (1 + self.a_for_erf * x * x)
        return torch.sign(x) * torch.sqrt(1 - torch.exp(exp))

    def erfinv_approx(self, x):
        b = -2 / (math.pi * self.a_for_erf) - torch.log(1 - x * x) / 2
        return torch.sign(x) * torch.sqrt(b + torch.sqrt(b * b - torch.log(1 - x * x) / self.a_for_erf))

    def forward(self, x):
        return (1 + self.erf_approx(x.mul(math.sqrt(0.5)))) / 2

    def backward(self, x):
        """
        The derivative of the standard Normal CDF is the standard normal PDF
        """
        normalizing_constant = 1.0 / math.sqrt(2 * math.pi)
        return x.pow(2).div_(-2).exp_().mul_(normalizing_constant)
