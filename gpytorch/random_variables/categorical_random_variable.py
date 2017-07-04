import torch
from torch.autograd import Variable
import random
import math
from .random_variable import RandomVariable

class CategoricalRandomVariable(RandomVariable):
    def __init__(self, mass_function):
        if isinstance(mass_function, Variable):
            mass_function = mass_function.data

        self.mass_function = mass_function
        self._cumulative_mass_function = self.mass_function.cumsum(0)

    def representation(self):
        return self.mass_function

    def log_probability(self, i):
        if i > len(self.mass_function):
            raise RuntimeError('Attempted to access a Categorical mass function with a category number larger than the total number of categories: %d'.format(i))

        return math.log(self.mass_function[i])

    def sample(self):
        p = random.random()
        cmf_lt = self._cumulative_mass_function.ge(p)
        for i,v in enumerate(cmf_lt):
            if v == 1:
                return torch.LongTensor([i])

    def num_categories(self):
        return len(self.mass_function)
