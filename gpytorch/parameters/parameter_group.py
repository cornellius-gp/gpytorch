from torch.nn import Parameter
from torch.autograd import Variable


class ParameterGroup(object):
    def __init__(self, **kwargs):
        for name, param in kwargs.items():
            setattr(self, name, param)
        self._options = {}


    def initialize(self, **kwargs):
        for param_name, param_value in kwargs.items():
            if hasattr(self, param_name):
                if isinstance(param_value, torch.Tensor):
                    getattr(self, param_name).data.copy_(param_value)
                else:
                    getattr(self, param_name).data.fill_(param_value)
            else:
                raise Exception('%s has no parameter %s' % (self.__class__.__name__, param_name))
        return self


    def set_options(self, **kwargs):
        for name, val in kwargs.items():
            self._options[name] = val


    def update(self,loss_closure):
        raise NotImplementedError

    def has_converged(self,loss_closure):
        raise NotImplementedError

    def __len__(self):
        return len(self.param_dict())


    def __iter__(self):
        for name in self.__dict__:
            value = getattr(self, name)
            if isinstance(value, Variable):
                yield name, value