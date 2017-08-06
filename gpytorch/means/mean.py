import torch
import gpytorch


class Mean(gpytorch.Module):
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

    def forward(self, x):
        raise NotImplementedError()
