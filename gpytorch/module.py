#!/usr/bin/env python3

import itertools
import warnings
from collections import OrderedDict

import torch
from torch import nn
from torch.distributions import Distribution

from .lazy import LazyTensor
from .utils.deprecation import DeprecationError


class Module(nn.Module):
    def __init__(self):
        super().__init__()
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
        # TODO: Change to initialize actual parameter (e.g. lengthscale) rather than untransformed parameter.
        """
        Set a value for a parameter

        kwargs: (param_name, value) - parameter to initialize
        Value can take the form of a tensor, a float, or an int
        """
        from .utils.log_deprecation import MODULES_WITH_LOG_PARAMS

        for name, val in kwargs.items():
            if isinstance(val, int):
                val = float(val)
            if any(isinstance(self, mod_type) for mod_type in MODULES_WITH_LOG_PARAMS) and "log_" in name:
                base_name = name.split("log_")[1]
                name = "raw_" + base_name
                if not torch.is_tensor(val):
                    val = self._inv_param_transform(torch.tensor(val).exp()).item()
                else:
                    val = self._inv_param_transform(val.exp())

            if not hasattr(self, name):
                raise AttributeError("Unknown parameter {p} for {c}".format(p=name, c=self.__class__.__name__))
            elif name not in self._parameters:
                setattr(self, name, val)
            elif torch.is_tensor(val):
                try:
                    self.__getattr__(name).data.copy_(val.expand_as(self.__getattr__(name)))
                except RuntimeError:
                    self.__getattr__(name).data.copy_(val.view_as(self.__getattr__(name)))

            elif isinstance(val, float):
                self.__getattr__(name).data.fill_(val)
            else:
                raise AttributeError("Type {t} not valid for initializing parameter {p}".format(t=type(val), p=name))

            # Ensure value is contained in support of prior (if present)
            prior_name = "_".join([name, "prior"])
            if prior_name in self._priors:
                prior, closure, _ = self._priors[prior_name]
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
        super().register_parameter(name, parameter)

    def register_prior(self, name, prior, param_or_closure, setting_closure=None):
        """
        Adds a prior to the module. The prior can be accessed as an attribute using the given name.

        Args:
            :attr:`name` (str):
                The name of the prior
            :attr:`prior` (Prior):
                The prior to be registered`
            :attr:`param_or_closure` (string or callable):
                Either the name of the parameter, or a closure (which upon calling evalutes a function on
                one or more parameters):
                single parameter without a transform: `.register_prior("foo_prior", foo_prior, "foo_param")`
                transform a single parameter (e.g. put a log-Normal prior on it):
                `.register_prior("foo_prior", NormalPrior(0, 1), lambda: torch.log(self.foo_param))`
                function of multiple parameters:
                `.register_prior("foo2_prior", foo2_prior, lambda: f(self.param1, self.param2)))`
            :attr:`setting_closure` (callable, optional):
                A function taking in a tensor in (transformed) parameter space and initializing the
                internal parameter representation to the proper value by applying the inverse transform.
                Enables setting parametres directly in the transformed space, as well as sampling
                parameter values from priors (see `sample_from_prior`)

        """
        if isinstance(param_or_closure, str):
            if param_or_closure not in self._parameters:
                raise AttributeError(
                    "Unknown parameter {name} for {module}".format(
                        name=param_or_closure, module=self.__class__.__name__
                    )
                    + "Make sure the parameter is registered before registering a prior."
                )

            def closure():
                return self._parameters[param_or_closure]

            if setting_closure is not None:
                raise RuntimeError("Must specify a closure instead of a parameter name when providing setting_closure")

            def setting_closure(val):
                return self.initialize(**{param_or_closure: val})

        else:
            closure = param_or_closure
        self.add_module(name, prior)
        self._priors[name] = (prior, closure, setting_closure)

    def sample_from_prior(self, prior_name):
        """Sample parameter values from prior. Modifies the module's parameters in-place."""
        if prior_name not in self._priors:
            raise RuntimeError("Unknown prior name '{}'".format(prior_name))
        prior, _, setting_closure = self._priors[prior_name]
        if setting_closure is None:
            raise RuntimeError("Must provide inverse transform to be able to sample from prior.")
        setting_closure(prior.sample())

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

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        from .utils.log_deprecation import LOG_DEPRECATION_MSG, MODULES_WITH_LOG_PARAMS

        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if not any(isinstance(self, mod_type) for mod_type in MODULES_WITH_LOG_PARAMS):
            return

        # Load log space parameters and throw deprecation warnings.
        for name, param in local_state.items():
            if "raw_" in name:
                base_name = name.split("raw_")[1]
                log_name = "log_" + base_name
                log_key = prefix + log_name
                if log_key in state_dict and log_key not in local_state:
                    warnings.warn(LOG_DEPRECATION_MSG.format(log_name=log_name, name=name), DeprecationWarning)
                    input_param = state_dict[log_key]
                    if isinstance(input_param, nn.Parameter):
                        input_param = input_param.data

                    real_input_param = self._inv_param_transform(input_param.exp())
                    param.copy_(real_input_param)

                    if prefix + name in missing_keys:
                        missing_keys.remove(prefix + name)
                    if prefix + name in unexpected_keys:
                        unexpected_keys.remove(prefix + log_name)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            from .utils.log_deprecation import LOG_DEPRECATION_MSG, MODULES_WITH_LOG_PARAMS

            if any(isinstance(self, mod_type) for mod_type in MODULES_WITH_LOG_PARAMS) and "log_" in name:
                base_name = name.split("log_")[1]  # e.g. log_lengthscale -> lengthscale
                raw_name = "raw_" + base_name
                warnings.warn(LOG_DEPRECATION_MSG.format(log_name=name, name=raw_name), DeprecationWarning)
                return super().__getattribute__(base_name).log()  # Get real param value and transform to log
            else:
                try:
                    return super().__getattribute__(name)
                except AttributeError:
                    raise e


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
        for name, (prior, closure, inv_closure) in module._priors.items():
            if prior is not None and prior not in memo:
                memo.add(prior)
                full_name = ("." if prefix else "").join([prefix, name])
                yield full_name, prior, closure, inv_closure
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, prior, closure, inv_closure in _extract_named_priors(module_, memo=memo, prefix=submodule_prefix):
            yield name, prior, closure, inv_closure
