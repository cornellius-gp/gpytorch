import torch
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
from .random_variables import RandomVariable
from .lazy import LazyVariable


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._bounds = OrderedDict()

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        for input in inputs:
            if not(isinstance(input, RandomVariable) or isinstance(input, Variable)):
                raise RuntimeError('Input must be a RandomVariable or Variable, was a %s' %
                                   input.__class__.__name__)
        outputs = self.forward(*inputs, **kwargs)
        if isinstance(outputs, Variable) or isinstance(outputs, RandomVariable) or isinstance(outputs, LazyVariable):
            return outputs

        for output in outputs:
            if not (isinstance(output, RandomVariable) or
                    isinstance(output, Variable) or
                    isinstance(output, LazyVariable)):
                raise RuntimeError('Output must be a RandomVariable, Variable, or LazyVariable. Was a %s' %
                                   input.__class__.__name__)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def register_parameter(self, name, param, bounds, prior=None):
        """
        Adds a parameter to the module.
        The parameter can be accessed as an attribute using given name.

        name (str): name of parameter
        param (torch.nn.Parameter): parameter
        bounds (2-tuple of float or Tensor): lower and upper bounds for parameter
        prior (RandomVariable): prior for parameter (default: None)
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        super(Module, self).register_parameter(name, param)

        # Set bounds
        lower_bound, upper_bound = bounds
        if torch.is_tensor(lower_bound) and torch.is_tensor(upper_bound):
            if lower_bound.size() != upper_bound.size() or \
                    lower_bound.size() != param.size():
                raise AttributeError('Lower bound, upper bound, and param should have the same size')
            self._bounds[name] = (lower_bound, upper_bound)
        elif (isinstance(lower_bound, int) or isinstance(lower_bound, float)) and \
                (isinstance(upper_bound, int) or isinstance(upper_bound, float)):
            lower_bound_tensor = param.data.new().resize_as_(param.data).fill_(lower_bound)
            upper_bound_tensor = param.data.new().resize_as_(param.data).fill_(upper_bound)
            self._bounds[name] = (lower_bound_tensor, upper_bound_tensor)
        else:
            raise AttributeError('Unsupported argument types for parameter %s' % name)

    def bound_for(self, name):
        if '.' in name:
            module, name = name.split('.', 1)
            if module in self._modules:
                return self.__getattr__(module).bound_for(name)
            else:
                raise AttributeError('Invalid bound name %s. '
                                     '%s has no module %s' % (name, type(self).__name__, module))
        else:
            if name in self._parameters:
                return self._bounds[name]
            else:
                raise AttributeError('Invalid bound name %s. '
                                     '%s has no parameter %s' % (name, type(self).__name__, module))


    def named_parameter_bounds(self):
        """
        Returns an iterator over module parameters bounds, yielding both the
        name of the parameter as well as the parameter bound itself
        """
        for name, _ in self.named_parameters():
            yield name, self.bound_for(name)

    def parameter_bounds(self):
        """
	Returns an iterator over module parameters bounds.
        This is typically passed to an optimizer.
        """
        for name, bound in self.named_parameter_bounds():
            yield bound

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                param = _parameters[name]
                # Ensure parameter is within bounds
		lower_bound, upper_bound = self._bounds[name]
                lower_mask = param.data < lower_bound
                if any(lower_mask.view(-1)):
                    param.data.masked_scatter_(lower_mask, lower_bound[lower_mask])
                upper_mask = param.data > upper_bound
                if any(upper_mask.view(-1)):
                    param.data.masked_scatter_(upper_mask, upper_bound[upper_mask])
                return param
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
	if isinstance(value, nn.Parameter):
            raise RuntimeError("Please assign torch.nn.Parameters using"
                               "gpytorch.module.register_parameters()")
        else:
            super(Module, self).__setattr__(name, value)
