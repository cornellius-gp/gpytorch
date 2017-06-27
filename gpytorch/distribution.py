import torch
from torch.autograd import Variable
from torch.nn import Module
from gpytorch.random_variables import RandomVariable, GaussianRandomVariable


class Distribution(Module):
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


    def __call__(self, *inputs, **kwargs):
        for input in inputs:
            assert(isinstance(input, Distribution) or isinstance(input, Variable),
                    'Input must be a Distribution or RandomVariable, was a %s' % \
                    input.__class__.__name__)
        outputs = self.forward(*inputs)
        if isinstance(outputs, Variable) or isinstance(outputs, RandomVariable):
            return outputs

        for output in outputs:
            assert(isinstance(output, RandomVariable) or isinstance(output, Variable),
                    'Output must be a Distribution or RandomVariable, was a %s' % \
                    input.__class__.__name__)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

