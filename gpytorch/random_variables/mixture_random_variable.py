from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from .random_variable import RandomVariable
from torch.autograd import Variable


class MixtureRandomVariable(RandomVariable):
    def __init__(self, *rand_vars, **kwargs):
        """
        Mixture of random variables

        Params:
        - rand_vars (iterable of RandomVariables)
        - weights (Variable or Tensor) weights of each of the random variables

        Note that weights must sum to 1
        """
        super(MixtureRandomVariable, self).__init__(*rand_vars, **kwargs)
        if not all(isinstance(rand_var, RandomVariable) for rand_var in rand_vars):
            raise RuntimeError("Everything needs to be an instance of a random variable")

        weights = kwargs.get("weights", None)
        if weights is None:
            weights = rand_vars[0].representation()[0].data.new(len(rand_vars)).fill_(1. / len(rand_vars))

        if torch.is_tensor(weights):
            weights = Variable(weights)

        if math.fabs(sum(weights.data) - 1) > 1e-4:
            raise RuntimeError("Weights must sum to 1")

        self.rand_vars = rand_vars
        self.weights = weights

    def mean(self):
        means = [rand_var.mean() for rand_var in self.rand_vars]
        return sum(weight.expand_as(mean) * mean for weight, mean in zip(self.weights, means))

    def representation(self):
        return self.rand_vars, self.weights

    def var(self):
        overall_mean = self.mean()
        means = [rand_var.mean() for rand_var in self.rand_vars]
        variances = [rand_var.var() for rand_var in self.rand_vars]
        return sum(
            weight.expand_as(mean) * ((mean - overall_mean) ** 2 + variance)
            for weight, mean, variance in zip(self.weights, means, variances)
        )
