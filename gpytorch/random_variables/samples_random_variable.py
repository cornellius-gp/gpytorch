from torch.autograd import Variable
from .random_variable import RandomVariable
import random


class SamplesRandomVariable(RandomVariable):
    def __init__(self, samples):
        """
        Constructs a random variable from samples

        samples should be a Variable of tensor, with the first dimension representing the samples

        Params:
        - samples (Variable: b x ...) samples
        """
        super(SamplesRandomVariable, self).__init__(samples)
        if not isinstance(samples, Variable):
            raise RuntimeError('samples should be a Variable')
        self._samples = samples

    def sample(self):
        ix = random.randrange(len(self._sample_list))
        return self._samples.data[ix]

    def representation(self):
        return self._samples
