from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import torch
from torch import nn
from torch.distributions import Distribution

from .lazy import LazyTensor
from .utils.deprecation import DeprecationError


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._added_loss_terms = OrderedDict()
        self._priors = OrderedDict()

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

    def _get_module_and_name(self, parameter_name):
        """Get module and name from full parameter name."""
        module, name = parameter_name.split(".", 1)
        if module in self._modules:
            return self.__getattr__(module), name
        else:
            raise AttributeError(
                "Invalid parameter name {}. {} has no module {}".format(parameter_name, type(self).__name__, module)
            )

    def added_loss_terms(self):
        for _, strategy in self.named_added_loss_terms():
            yield strategy

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def hyperparameters(self):
        for _, param in self.named_hyperparameters():
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
            prior_name = "_".join([name, "prior"])
            if prior_name in self._priors:
                prior, closure = self._priors[prior_name]
                try:
                    prior._validate_sample(closure())
                except ValueError as e:
                    raise ValueError("Invalid input value for prior {}. Error:\n{}".format(prior_name, e))

        return self

    def named_added_loss_terms(self):
        """Returns an iterator over module variational strategies, yielding both
        the name of the variational strategy as well as the strategy itself.

        Yields:
            (string, VariationalStrategy): Tuple containing the name of the
                strategy and the strategy

        """
        return _extract_named_added_loss_terms(module=self, memo=None, prefix="")

    def named_hyperparameters(self):
        for name, param in self.named_parameters():
            if "variational_" not in name:
                yield name, param

    def named_priors(self, memo=None, prefix=""):
        """Returns an iterator over the module's priors, yielding the name of the prior,
        the prior, the associated parameter names, and the transformation callable.

        Yields:
            (string, Prior, tuple((Parameter, callable)), callable): Tuple containing:
                - the name of the prior
                - the prior
                - a tuple of tuples (param, transform), one for each of the parameters associated with the prior
                - the prior's transform to be called on the parameters
        """
        return _extract_named_priors(module=self, memo=None, prefix="")

    def named_variational_parameters(self):
        for name, param in self.named_parameters():
            if "variational_" in name:
                yield name, param

    def register_added_loss_term(self, name):
        self._added_loss_terms[name] = None

    def register_parameter(self, name, parameter, prior=None):
        r"""
        Adds a parameter to the module. The parameter can be accessed as an attribute using the given name.

        Args:
            :attr:`name` (str):
                The name of the parameter
            :attr:`parameter` (torch.nn.Parameter):
                The parameter
        """
        if prior is not None:
            raise DeprecationError(
                "Setting a prior upon registering a parameter is deprecated. Please use "
                ".register_prior('{name}_prior', prior, '{name}') instead.".format(name=name)
            )
        if "_parameters" not in self.__dict__:
            raise AttributeError("Cannot assign parameter before Module.__init__() call")
        super(Module, self).register_parameter(name, parameter)

    def register_prior(self, name, prior, arg):
        """
        Adds a prior to the module. The prior can be accessed as an attribute using the given name.

        Args:
            :attr:`name` (str):
                The name of the prior
            :attr:`prior` (Prior):
                The prior to be registered`
            :attr:`arg` (string or callable):
                Either the name of the parameter, or a closure (which upon calling evalutes a function on
                    one or more parameters):
                - single parameter without a transform: `.register_prior("foo_prior", foo_prior, "foo_param")`
                - transform single parameter (e.g. put a log-Normal prior on it):
                    `.register_prior("foo_prior", NormalPrior(0, 1), lambda: torch.log(self.foo_param))`
                - function of multiple parameters (e.g. put a prior over the ICM Kernel covariance matrix):
                    `.register_prior("foo2_prior", lkj_prior, lambda: _eval_covar_matrix(self.param1, self.param2)))`
        """
        if isinstance(arg, str):
            if arg not in self._parameters:
                raise AttributeError(
                    "Unknown parameter {name} for {module}".format(name=arg, module=self.__class__.__name__)
                    + "Make sure the parameter is registered before registering a prior."
                )

            def closure():
                return self._parameters[arg]

        else:
            closure = arg
        self.add_module(name, prior)
        self._priors[name] = (prior, closure)

    def update_added_loss_term(self, name, added_loss_term):
        from .mlls import AddedLossTerm

        if not isinstance(added_loss_term, AddedLossTerm):
            raise RuntimeError("added_loss_term must be a AddedLossTerm")
        if name not in self._added_loss_terms.keys():
            raise RuntimeError("added_loss_term {} not registered".format(name))
        self._added_loss_terms[name] = added_loss_term

    def variational_parameters(self):
        for _, param in self.named_variational_parameters():
            yield param


def _extract_named_added_loss_terms(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_added_loss_terms"):
        for name, strategy in module._added_loss_terms.items():
            if strategy is not None and strategy not in memo:
                memo.add(strategy)
                yield prefix + ("." if prefix else "") + name, strategy
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, strategy in _extract_named_added_loss_terms(module=module_, memo=memo, prefix=submodule_prefix):
            yield name, strategy


def _extract_named_priors(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_priors"):
        for name, (prior, closure) in module._priors.items():
            if prior is not None and prior not in memo:
                memo.add(prior)
                full_name = ("." if prefix else "").join([prefix, name])
                yield full_name, prior, closure
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, prior, closure in _extract_named_priors(module_, memo=memo, prefix=submodule_prefix):
            yield name, prior, closure
