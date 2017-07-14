import torch
from torch.nn import Module


class ParameterGroup(Module):
    def __init__(self):
        super(ParameterGroup, self).__init__()
        self._options = {}
        self._training = True

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

    def update(self, loss_closure):
        raise NotImplementedError

    def has_converged(self, loss_closure):
        raise NotImplementedError

    def toggle_training(self):
        self._training = not self._training
