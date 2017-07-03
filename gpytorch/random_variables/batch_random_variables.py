from torch.autograd import Variable
from .random_variable import RandomVariable
import torch

class BatchRandomVariables(RandomVariable):
    def __init__(self, random_variable, count):
        self.random_variable_list = [random_variable]*count

    def sample(self):
        return Variable(torch.cat([rv.sample().unsqueeze(0) for rv in self.random_variable_list]))

    def log_probability(self, x):
        return Variable(torch.Tensor([self.random_variable_list[0].log_probability(x)]))

    def __len__(self):
        return len(self.random_variable_list)

    def __iter__(self):
        for random_variable in self.random_variable_list:
            yield random_variable

    def __getitem__(self,i):
        return self.random_variable_list[i]