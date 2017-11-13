import torch
from torch.autograd import Variable
from .random_variable import RandomVariable


class CategoricalRandomVariable(RandomVariable):
    def __init__(self, mass_function):
        """
        Constructs a categorical random variable
        mass_function represents the weights of the distribution

        Passing a vector mass function corresponds to a single categorical variable
        Passing a matrix mass function corresponds to a batch of independent categorial
            variables.

        Params:
        - mass_function (Variable: vector k or matrix n x k) weights of categorical distribution
        """
        super(CategoricalRandomVariable, self).__init__(mass_function)
        if not isinstance(mass_function, Variable):
            raise RuntimeError('mass_function should be a Variable')

        ndimension = mass_function.ndimension()
        if ndimension not in [1, 2]:
            raise RuntimeError('mass_function should be a vector or a matrix')

        # Assert that probabilities sum to 1
        if ndimension == 1:
            mass_function = mass_function.unsqueeze(0)
        if torch.abs(mass_function.data.sum(1) - 1).gt(1e-5).sum():
            raise RuntimeError('mass_function probabilties (in each row) should sum to 1!')
        if ndimension == 1:
            mass_function = mass_function.squeeze(0)

        self.mass_function = mass_function

    def representation(self):
        return self.mass_function

    def sample(self, n_samples=1):
        mass_function = self.mass_function.data
        res = torch.multinomial(mass_function, n_samples, replacement=True)

        # Sample dimension is first
        if res.ndimension() == 2:
            res = res.t()
        return res

    def __len__(self):
        if self.mass_function.ndimension() == 2:
            return self.mass_function.size(0)
        return 1
