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
        kwargs = {}
        kwargs[name] = bounds
        self.set_bounds(**kwargs)

    def initialize(self, **kwargs):
        """
        Set a value for a parameter

        kwargs: (param_name, value) - parameter to initialize
        Value can take the form of a tensor, a float, or an int
        """
        for name, val in kwargs.items():
            if name not in self._parameters:
                raise AttributeError('Unknown parameter %s for %s' % (name, self.__class__.__name__))
            if torch.is_tensor(val):
                self.__getattr__(name).data.copy_(val)
            elif isinstance(val, float) or isinstance(val, int):
                self.__getattr__(name).data.fill_(val)
            else:
                raise AttributeError('Type %s not valid to initialize parameter %s' % (type(val), name))

            # Ensure initializion is within bounds
            param = self._parameters[name]
            lower_bound, upper_bound = self._bounds[name]
            lower_mask = param.data < lower_bound
            if any(lower_mask.view(-1)):
                raise AttributeError('Parameter %s exceeds lower bound' % name)
            upper_mask = param.data > upper_bound
            if any(upper_mask.view(-1)):
                raise AttributeError('Parameter %s exceeds upper bound' % name)
        return self

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        NOTE: This differs from the PyTorch load_state_dict.
        Buffer size doesn't have to match. This allows us to load caches

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].resize_as_(param).copy_(param)

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def set_bounds(self, **kwargs):
        """
        Set bounds for a parameter

        kwargs: (param_name, value) - parameter to initialize
        Value can take the form of a tensor, a float, or an int
        """
        for name, bounds in kwargs.items():
            if name not in self._parameters:
                raise AttributeError('Unknown parameter %s for %s' % (name, self.__class__.__name__))
            param = self._parameters[name]
            # Set bounds
            lower_bound, upper_bound = bounds
            if torch.is_tensor(lower_bound) and torch.is_tensor(upper_bound):
                if lower_bound.size() != upper_bound.size() or \
                        lower_bound.size() != param.size():
                    raise AttributeError('Lower bound, upper bound, and param should have the same size')
            elif not (isinstance(lower_bound, int) or isinstance(lower_bound, float)) or \
                    not (isinstance(upper_bound, int) or isinstance(upper_bound, float)):
                raise AttributeError('Unsupported argument types for parameter %s' % name)

            if name not in self._bounds:
                self._bounds[name] = [None, None]
            self._bounds[name][0] = lower_bound
            self._bounds[name][1] = upper_bound
        return self

    def bound_for(self, name):
        """
        Get bounds for parameter

        name (str): parameter name
        """
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

    def condition(self, *args, **kwargs):
        """
        Conditions the model on data. After conditioning, the model functions
        in posterior mode rather than prior mode.

        Each model class should override this method to do whatever needs to be
        done to the data. It is highly recommended that parent modules call this
        function on their children modules.

        Args: (Variables) inputs to condition on
        """
        if not all(isinstance(arg, Variable) or isinstance(arg, RandomVariable) for arg in args):
            raise RuntimeError('All inputs must be Variables')
        self.train_data = args
        return self

    @property
    def posterior(self):
        """
        Returns if the model is in posterior mode (are we conditioning on data?)
        """
        return hasattr(self, 'train_data') and not self.training

    def initialize_interpolation_grid(self, grid_size, grid_bounds):
        for module in self.children():
            module.initialize_interpolation_grid(grid_size, grid_bounds)
        return self

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                param = _parameters[name]
                # Ensure parameter is within bounds
                lower_bound, upper_bound = self._bounds[name]
                lower_mask = param.data < lower_bound
                if any(lower_mask.view(-1)):
                    if torch.is_tensor(lower_bound):
                        param.data.masked_scatter_(lower_mask, lower_bound[lower_mask])
                    else:
                        param.data.masked_fill_(lower_mask, lower_bound)
                upper_mask = param.data > upper_bound
                if any(upper_mask.view(-1)):
                    if torch.is_tensor(upper_bound):
                        param.data.masked_scatter_(upper_mask, upper_bound[upper_mask])
                    else:
                        param.data.masked_fill_(upper_mask, upper_bound)
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
