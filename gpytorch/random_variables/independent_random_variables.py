from .random_variable import RandomVariable
import torch
from torch.autograd import Variable


class IndependentRandomVariables(RandomVariable):
    def __init__(self, random_variable_list):
        self.random_variable_list = random_variable_list

    def log_probability(self, x):
        return Variable(torch.Tensor([sum([rv.log_probability() for rv in self.random_variable_list])]))

    def sample(self):
        return Variable(torch.cat([rv.sample().unsqueeze(0) for rv in self.random_variable_list]))

    def mean(self):
        return torch.cat([rv.mean().unsqueeze(0) for rv in self.random_variable_list])

    def __len__(self):
        return len(self.random_variable_list)

    def __iter__(self):
        for random_variable in self.random_variable_list:
            yield random_variable

    def __getitem__(self, i):
        return self.random_variable_list[i]
