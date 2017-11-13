import torch
from torch.autograd import Variable
from .random_variable import RandomVariable


class BernoulliRandomVariable(RandomVariable):
    def __init__(self, probability):
        """
        Constructs a Bernoulli random variable
        probability represents the weight of a positive outcome

        Passing a scalar mass function corresponds to a single Bernoulli variable
        Passing a vector mass function corresponds to a batch of independent Bernoulli variables

        Params:
        - probability (Variable: scalar or vector n) weights of Bernoulli distribution
        """
        super(BernoulliRandomVariable, self).__init__(probability)
        if not isinstance(probability, Variable):
            raise RuntimeError('probability should be a Variable')
        if not probability.ndimension() == 1:
            raise RuntimeError('BernoulliRandomVariable should be a scalar or a vector')
        if probability.data.lt(0).sum() or probability.data.gt(1).sum():
            raise RuntimeError('Probabilities must be between 0 and 1')
        self.probability = probability

    def representation(self):
        return self.probability

    def sample(self, n_samples=1):
        prob = torch.rand(n_samples, len(self.probability))
        res = prob < self.probability.data.unsqueeze(0).expand(n_samples, len(self.probability))
        if len(self.probability) == 1:
            res.squeeze_(1)
        return res

    def mean(self):
        return self.probability

    def __len__(self):
        return len(self.probability)
