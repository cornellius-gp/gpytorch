from .prior import Prior
import random
import math

class CategoricalPrior(Prior):
    def __init__(self,mass_function):
        self.mass_function = mass_function
        self._cumulative_mass_function = self.mass_function.cumsum()


    def forward(self, i):
        if i > len(self.mass_function):
            raise RuntimeError('Attempted to access a Categorical mass function with a category number larger than the total number of categories: %d'.format(i))

        return math.log(self.mass_function[i])


    def sample(self):
        p = random.random()
        cmf_lt = self._cumulative_mass_function.ge(p)
        for i,v in enumerate(cmf_lt):
            if v == 1:
                return i

    def num_categories(self):
        return len(self.mass_function)