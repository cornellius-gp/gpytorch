from torch.autograd import Variable
from .random_variable import RandomVariable
import random

class SamplesRandomVariable(RandomVariable):
    def __init__(self, sample_list):
        self._sample_list = sample_list

    def sample(self):
        ix = random.randrange(len(self._sample_list))
        return Variable(self._sample_list[ix])

    def __setitem__(self, key, value):
        self._sample_list[key] = value

    def __getitem__(self,key):
        return self._sample_list[key]