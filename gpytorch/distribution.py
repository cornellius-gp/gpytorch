import torch
from torch.autograd import Variable
from torch.nn import Module, Parameter
from gpytorch.random_variables import RandomVariable, GaussianRandomVariable
from gpytorch.parameters import ParameterGroup


class Distribution(Module):
    def __init__(self):
        super(Distribution, self).__init__()
        self._parameter_groups = {}


    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


    def __call__(self, *inputs, **kwargs):
        for input in inputs:
            assert(isinstance(input, Distribution) or isinstance(input, Variable),
                    'Input must be a Distribution or RandomVariable, was a %s' % \
                    input.__class__.__name__)
        outputs = self.forward(*inputs, **kwargs)
        if isinstance(outputs, Variable) or isinstance(outputs, RandomVariable):
            return outputs

        for output in outputs:
            assert(isinstance(output, RandomVariable) or isinstance(output, Variable),
                    'Output must be a Distribution or RandomVariable, was a %s' % \
                    input.__class__.__name__)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def __setattr__(self,name,value):
        if isinstance(value,ParameterGroup):
            self._parameter_groups[name] = value
        elif isinstance(value,Parameter):
            raise RuntimeError('Observation Models expect ParameterGroups, not Parameters directly.')
        else:
            super(Distribution, self).__setattr__(name, value)


    def __getattr__(self, name):
        if name in self._parameter_groups:
            return self._parameter_groups[name]
        else:
            return super(Distribution, self).__getattr__(name)


    def parameter_groups(self):
        for name, param_group in self.named_parameter_groups():
            yield param_group


    def named_parameter_groups(self):
        for name, param_group in self._parameter_groups.items():
            yield name, param_group

        for child in self.children():
            if isinstance(child,Distribution):
                for name, param_group in child.named_parameter_groups():
                    yield name, param_group

