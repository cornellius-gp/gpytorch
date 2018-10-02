from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import torch
from torch import nn
from torch.distributions import Distribution

from .lazy import LazyTensor
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
                "Invalid parameter name {}. {} has no module {}".format(parameter_name, type(self).__name__, module)
            )

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def hyperparameters(self):
        for name, param in self.named_hyperparameters():
            yield param

    def initialize(self, **kwargs):
        """
        Set a value for a parameter

        kwargs: (param_name, value) - parameter to initialize
        Value can take the form of a tensor, a float, or an int
        """
        for name, val in kwargs.items():
            if name not in self._parameters:
                raise AttributeError("Unknown parameter {p} for {c}".format(p=name, c=self.__class__.__name__))
            if torch.is_tensor(val):
                self.__getattr__(name).data.copy_(val)
            elif isinstance(val, float) or isinstance(val, int):
                self.__getattr__(name).data.fill_(val)
            else:
                raise AttributeError("Type {t} not valid to initialize parameter {p}".format(t=type(val), p=name))

            # Ensure value is contained in support of prior (if present)
            prior = self._priors.get(name)
            if prior is not None:
                param = self._parameters[name]
                try:
                    prior._validate_sample(param)
                except ValueError as e:
                    raise ValueError(
                        "Value of parameter {p} not valid for specified prior. Original exception:\n{e}".format(
                            p=param, e=e
                        )
                    )
        return self

    def named_parameter_priors(self):
        """
        Returns an iterator over module parameter priors, yielding the name of
        the parameter, the parameter itself, as well as the associated prior
        (excludes parameters for which no prior has been registered)
        """
        return _extract_named_parameter_priors(module=self, memo=None, prefix="")

    def named_derived_priors(self, memo=None, prefix=""):
        """Returns an iterator over module derived priors, yielding both the
        name of the prior as well as the prior, the associated parameters, and
        the transformation callable.

        Yields:
            (string, Prior, tuple(string), callable): Tuple containing the name
                of the prior, the prior itself, its parameters, and the transform
                to be called on the parameters.

        """
        return _extract_named_derived_priors(module=self, memo=None, prefix="")

    def named_hyperparameters(self):
        for name, param in self.named_parameters():
            if "variational_" not in name:
                yield name, param

    def named_variational_parameters(self):
        for name, param in self.named_parameters():
            if "variational_" in name:
                yield name, param

    def named_variational_strategies(self):
        """Returns an iterator over module variational strategies, yielding both
        the name of the variational strategy as well as the strategy itself.

        Yields:
            (string, VariationalStrategy): Tuple containing the name of the
                strategy and the strategy

        """
        return _extract_named_variational_strategies(module=self, memo=None, prefix="")

    def register_parameter(self, name, parameter, prior=None):
        """
        Adds a parameter to the module.
        The parameter can be accessed as an attribute using given name.

        name (str): name of parameter
        param (torch.nn.Parameter): parameter
        prior (Prior): prior for parameter (default: None)
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError("Cannot assign parameter before Module.__init__() call")
        super(Module, self).register_parameter(name, parameter)
        if prior is not None:
            self.set_parameter_priors(**{name: prior})

    def register_derived_prior(self, name, prior, parameter_names, transform):
        """
        Adds a derived prior to the module.
        The prior can be accessed as an attribute using the given name.

        name (str): name of the derived prior
        prior (Prior): the prior object
        parameter_names (tuple(str)): The parameters the transform operaters on,
            in the same order as expected by the transform callable.
        transform (Callable): The function called on the specified parameters. The
            log-pdf of the prior will be evaluating on the output of this transform.

        A derived prior operates on a transform of one or multiple parameters.
        This can be used, for instance, to put a prior over the ICM Kernel
        covariance matrix generated from covar_factor and log_var parameters.

        """
        self.add_module(name, prior)
        self._derived_priors[name] = (prior, tuple(parameter_names), transform)

    def register_variational_strategy(self, name):
        self._variational_strategies[name] = None

    def set_parameter_priors(self, **kwargs):
        """
        Set prior for a parameter.
        The prior can be accessed as an attribute using <PARAMETER_NAME>_prior.

        kwargs: (param_name, prior) - parameter to initialize
        prior must be a gpytorch Prior
        """
        for name, prior in kwargs.items():
            if name not in self._parameters:
                raise AttributeError(
                    "Unknown parameter {name} for {module}".format(name=name, module=self.__class__.__name__)
                )
            self.add_module("_".join([name, "prior"]), prior)
            self._priors[name] = prior
        return self

    def variational_parameters(self):
        for name, param in self.named_variational_parameters():
            yield param

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

        if isinstance(outputs, tuple):
            if not all(
                torch.is_tensor(output) or isinstance(output, Distribution) or isinstance(output, LazyTensor)
                for output in outputs
            ):
                raise RuntimeError(
                    "All outputs must be a Distribution, torch.Tensor, or LazyTensor. "
                    "Got {}".format([output.__class__.__name__ for output in outputs])
                )
            if len(outputs) == 1:
                outputs = outputs[0]
            return outputs

        elif torch.is_tensor(outputs) or isinstance(outputs, Distribution) or isinstance(outputs, LazyTensor):
            return outputs
        else:
            raise RuntimeError(
                "Output must be a Distribution, torch.Tensor, or LazyTensor. Got {}".format(outputs.__class__.__name__)
            )


def _extract_named_parameter_priors(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_priors"):
        for name, parameter in module._parameters.items():
            if name in module._priors and module._priors[name] not in memo:
                prior = module._priors[name]
                memo.add(prior)
                yield prefix + ("." if prefix else "") + name, parameter, prior
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, parameter, prior in _extract_named_parameter_priors(
            module=module_, memo=memo, prefix=submodule_prefix
        ):
            yield name, parameter, prior


def _extract_named_variational_strategies(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_variational_strategies"):
        for name, strategy in module._variational_strategies.items():
            if strategy is not None and strategy not in memo:
                memo.add(strategy)
                yield prefix + ("." if prefix else "") + name, strategy
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, strategy in _extract_named_variational_strategies(module=module_, memo=memo, prefix=submodule_prefix):
            yield name, strategy


def _extract_named_derived_priors(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_derived_priors"):
        for name, (prior, pnames, tf) in module._derived_priors.items():
            if prior is not None and prior not in memo:
                memo.add(prior)
                parameters = tuple(getattr(module, pname) for pname in pnames)
                yield prefix + ("." if prefix else "") + name, prior, parameters, tf
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, prior, parameters, tf in _extract_named_derived_priors(module_, memo=memo, prefix=submodule_prefix):
            yield name, prior, parameters, tf
