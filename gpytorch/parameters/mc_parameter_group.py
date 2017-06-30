from torch.autograd import Variable
from .parameter_group import ParameterGroup
from priors import Prior

class MCParameterGroup(ParameterGroup):
    def __init__():
        self._training = True
        self._priors = {}
        self._options = {'num_samples': 20}

        for name, param in kwargs.items():
            var, prior = param
            if not isinstance(prior, Prior):
                raise RuntimeError('All parameters in an MCParameterGroup must have Priors')

            if not isinstance(var, Variable):
                raise RuntimeError('All parameters in an MCParameterGroup must have an associated Variable')

            setattr(self,name,var)
            self._priors[name] = prior
            self._samples[name] = []


    def sample(self):
        raise NotImplementedError

    def has_converged(self,loss_closure):
        return True


