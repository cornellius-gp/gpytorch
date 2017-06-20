import torch
from torch.nn import Module

class Kernel(Module):
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


    def forward(self, x1, x2):
        raise NotImplementedError()


    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        if x1.data.ndimension() == 1:
            x1 = x1.view(-1, 1)
        if x2.data.ndimension() == 1:
            x2 = x2.view(-1, 1)
        assert(x1.size(1) == x2.size(1))
        return super(Kernel, self).__call__(x1, x2)
