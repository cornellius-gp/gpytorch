import torch
import math
from .random_variable import RandomVariable
import random


class BernoulliRandomVariable(RandomVariable):
    def __init__(self, probability):
        self.probability = probability

    def representation(self):
        return self.probability

    def log_probability(self, i):
        if i == 1:
            return math.log(self.probability)
        elif i == -1:
            return math.log(1 - self.probability)
        else:
            raise RuntimeError('The Bernoulli probability mass function is defined on {-1,1}')

    def sample(self):
        p = random.random()
        if p < self.probability:
            return torch.LongTensor([1])
        else:
            return torch.LongTensor([-1])

    def mean(self):
        return self.probability
