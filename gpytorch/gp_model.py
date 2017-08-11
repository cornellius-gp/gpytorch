import gpytorch
from torch.autograd import Variable
from .random_variables import RandomVariable
from .lazy import LazyVariable


class GPModel(gpytorch.Module):
    def __init__(self, likelihood):
        super(GPModel, self).__init__()
        self._parameter_groups = {}
        self.likelihood = likelihood

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        output = super(GPModel, self).__call__(*args, **kwargs)
        if isinstance(output, Variable) or isinstance(output, RandomVariable) or isinstance(output, LazyVariable):
            output = (output,)
        return self.likelihood(*output)
