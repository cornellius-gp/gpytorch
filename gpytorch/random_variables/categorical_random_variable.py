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
        """
        This function is written as a reparameterization
        This way we can differentiate through the mass function
        """
        mass_function = self.mass_function
        if mass_function.ndimension() == 1:
            mass_function = mass_function.unsqueeze(0)

        upper_mass_function = mass_function.cumsum(1)
        lower_mass_function = upper_mass_function - mass_function

        # Generate uniform samples
        samples = Variable(mass_function.data.new(n_samples, 1, 1).uniform_())
        samples = samples.clamp(1e-5, 1)  # Make sure that everything is strictly greater than zero

        lower_mask = samples.gt(lower_mass_function.unsqueeze(0))
        upper_mask = samples.le(upper_mass_function.unsqueeze(0))
        res = (lower_mask * upper_mask)

        # Sample dimension is first
        if self.mass_function.ndimension() == 1:
            res = res.squeeze(1)
        if n_samples == 1:
            res = res.squeeze(0)
        return res

    def __len__(self):
        if self.mass_function.ndimension() == 2:
            return self.mass_function.size(0)
        return 1
