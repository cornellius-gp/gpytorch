from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from collections import OrderedDict
from torch import nn
from .random_variables import RandomVariable
from .lazy import LazyVariable
from .variational import VariationalStrategy


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._priors = OrderedDict()
        self._derived_priors = OrderedDict()
        self._variational_strategies = OrderedDict()

    def _get_module_and_name(self, parameter_name):
        """Get module and name from full parameter name."""
        module, name = parameter_name.split(".", 1)
        if module in self._modules:
            return self.__getattr__(module), name
        else:
            raise AttributeError(
                "Invalid parameter name {}. {} has no module {}".format(
                    parameter_name, type(self).__name__, module
                )
            )

    def _get_prior_for(self, parameter_name):
        """
        Get prior for parameter

        parameter_name (str): parameter name
        """
        if "." in parameter_name:
            module, parameter_name = self._get_module_and_name(parameter_name)
            return module._get_prior_for(parameter_name)
        else:
            if parameter_name in self._parameters:
                return self._priors.get(parameter_name)
            else:
                raise AttributeError(
                    "Module {module} has no parameter {name}".format(
                        module=type(self).__name__, name=parameter_name
                    )
                )

    def _get_derived_prior(self, prior_name):
        """
        Get derived prior from prior name

        prior_name (str): the name of the derived prior
        """
        if "." in prior_name:
            module, prior_name = self._get_module_and_name(prior_name)
            return module._get_derived_prior(prior_name)
        else:
            if prior_name in self._parameters:
                return self._derived_priors.get(prior_name)
            else:
                raise AttributeError(
                    "Module {module} has no derived prior {name}".format(
                        module=type(self).__name__, name=prior_name
                    )
                )

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def initialize(self, **kwargs):
        """
        Set a value for a parameter

        kwargs: (param_name, value) - parameter to initialize
        Value can take the form of a tensor, a float, or an int
        """
        for name, val in kwargs.items():
            if name not in self._parameters:
                raise AttributeError(
                    "Unknown parameter {p} for {c}".format(
                        p=name, c=self.__class__.__name__
                    )
                )
            if torch.is_tensor(val):
                self.__getattr__(name).data.copy_(val)
            elif isinstance(val, float) or isinstance(val, int):
                self.__getattr__(name).data.fill_(val)
            else:
                raise AttributeError(
                    "Type {t} not valid to initialize parameter {p}".format(
                        t=type(val), p=name
                    )
                )

            # Ensure value is contained in support of prior (if present)
            prior = self._priors.get(name)
            if prior is not None:
                param = self._parameters[name]
                if not prior.is_in_support(param):
                    raise ValueError(
                        "Value of parameter {param} not contained in support "
                        "of specified prior".format(param=param)
                    )
        return self

    def named_parameter_priors(self):
        """
        Returns an iterator over module parameter priors, yielding the name of
        the parameter, the parameter itself, as well as the associated prior
        (excludes parameters for which no prior has been registered)
        """
        for name, param in self.named_parameters():
            prior = self._get_prior_for(name)
            if prior is not None:
                yield name, param, prior

    def named_derived_priors(self, memo=None, prefix=""):
        """Returns an iterator over module derived priors, yielding both the
        name of the prior as well as the prior, the associated parameters, and
        the transformation callable.

        Yields:
            (string, Prior, tuple(string), callable): Tuple containing the name
                of the prior, the prior itself, its parameters, and the transform
                to be called on the parameters.

        """
        if memo is None:
            memo = set()
        for name, (prior, pnames, tf) in self._derived_priors.items():
            if prior is not None and prior not in memo:
                memo.add(prior)
                yield prefix + ("." if prefix else "") + name, prior, pnames, tf
        for mname, module in self.named_children():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            if hasattr(module, "_derived_priors"):
                for name, prior, pnames, tf in module.named_derived_priors(
                    memo, submodule_prefix
                ):
                    yield name, prior, pnames, tf

    def named_variational_strategies(self, memo=None, prefix=""):
        """Returns an iterator over module variational strategies, yielding both
        the name of the variational strategy as well as the strategy itself.

        Yields:
            (string, VariationalStrategy): Tuple containing the name of the
                strategy and the strategy

        """
        if memo is None:
            memo = set()
        for name, strategy in self._variational_strategies.items():
            if strategy is not None and strategy not in memo:
                memo.add(strategy)
                yield prefix + ("." if prefix else "") + name, strategy
        for mname, module in self.named_children():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            if hasattr(module, "named_variational_strategies"):
                for name, strategy in module.named_variational_strategies(
                    memo, submodule_prefix
                ):
                    yield name, strategy

    def register_parameter(self, name, parameter, prior=None):
        """
        Adds a parameter to the module.
        The parameter can be accessed as an attribute using given name.

        name (str): name of parameter
        param (torch.nn.Parameter): parameter
        prior (Prior): prior for parameter (default: None)
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "Cannot assign parameter before Module.__init__() call"
            )
        super(Module, self).register_parameter(name, parameter)
        if prior is not None:
            self.set_parameter_priors(**{name: prior})

    def register_derived_prior(self, name, prior, parameter_names, transform):
        """

        """
        self.add_module(name, prior)
        self._derived_priors[name] = (prior, tuple(parameter_names), transform)

    def register_variational_strategy(self, name):
        self._variational_strategies[name] = None

    def set_parameter_priors(self, **kwargs):
        """
        Set prior for a parameter

        kwargs: (param_name, prior) - parameter to initialize
        prior must be a gpytorch Prior
        """
        for name, prior in kwargs.items():
            if name not in self._parameters:
                raise AttributeError(
                    "Unknown parameter {name} for {module}".format(
                        name=name, module=self.__class__.__name__
                    )
                )
            self.add_module("_".join([name, "prior"]), prior)
            self._priors[name] = prior
        return self

    def variational_strategies(self):
        for _, strategy in self.named_variational_strategies():
            yield strategy

    def update_variational_strategy(self, name, variational_strategy):
        if not isinstance(variational_strategy, VariationalStrategy):
            raise RuntimeError("variational_strategy must be a VariationalStrategy")
        if name not in self._variational_strategies.keys():
            raise RuntimeError("variational strategy {} not registered".format(name))
        self._variational_strategies[name] = variational_strategy

    def __call__(self, *inputs, **kwargs):
        outputs = self.forward(*inputs, **kwargs)
        if (
            torch.is_tensor(outputs)
            or isinstance(outputs, RandomVariable)
            or isinstance(outputs, LazyVariable)
        ):
            return outputs
        for output in outputs:
            if not (
                isinstance(output, RandomVariable)
                or torch.is_tensor(output)
                or isinstance(output, LazyVariable)
            ):
                raise RuntimeError(
                    "Output must be a RandomVariable, torch.Tensor, or LazyVariable. "
                    "Was a {}".format(input.__class__.__name__)
                )
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def __getattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __setattr__(self, name, value):
        if isinstance(value, nn.Parameter):
            raise RuntimeError(
                "Please assign torch.nn.Parameters using"
                "gpytorch.module.register_parameters()"
            )
        super(Module, self).__setattr__(name, value)
