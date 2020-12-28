#!/usr/bin/env python3

import inspect
import itertools
from collections import OrderedDict

import torch
from torch import nn
from torch.distributions import Distribution

from .constraints import Interval
from .lazy import LazyTensor


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self._added_loss_terms = OrderedDict()
        self._priors = OrderedDict()
        self._constraints = OrderedDict()

        self._strict_init = True
        self._load_strict_shapes = True

        self._register_load_state_dict_pre_hook(self._load_state_hook_ignore_shapes)

    def __call__(self, *inputs, **kwargs):
        outputs = self.forward(*inputs, **kwargs)
        if isinstance(outputs, list):
            return [_validate_module_outputs(output) for output in outputs]
        return _validate_module_outputs(outputs)

    def _clear_cache(self):
        """
        Clear any precomputed caches.
        Should be implemented by any module that caches any computation at test time.
        """
        pass

    def _get_module_and_name(self, parameter_name):
        """Get module and name from full parameter name."""
        module, name = parameter_name.split(".", 1)
        if module in self._modules:
            return self.__getattr__(module), name
        else:
            raise AttributeError(
                "Invalid parameter name {}. {} has no module {}".format(parameter_name, type(self).__name__, module)
            )

    def _strict(self, value):
        _set_strict(self, value)

    def added_loss_terms(self):
        for _, strategy in self.named_added_loss_terms():
            yield strategy

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def constraints(self):
        for _, constraint in self.named_constraints():
            yield constraint

    def hyperparameters(self):
        for _, param in self.named_hyperparameters():
            yield param

    def initialize(self, **kwargs):
        """
        Set a value for a parameter

        kwargs: (param_name, value) - parameter to initialize.
        Can also initialize recursively by passing in the full name of a
        parameter. For example if model has attribute model.likelihood,
        we can initialize the noise with either
        `model.initialize(**{'likelihood.noise': 0.1})`
        or
        `model.likelihood.initialize(noise=0.1)`.
        The former method would allow users to more easily store the
        initialization values as one object.

        Value can take the form of a tensor, a float, or an int
        """

        for name, val in kwargs.items():
            if isinstance(val, int):
                val = float(val)
            if "." in name:
                module, name = self._get_module_and_name(name)
                module.initialize(**{name: val})
            elif not hasattr(self, name):
                raise AttributeError("Unknown parameter {p} for {c}".format(p=name, c=self.__class__.__name__))
            elif name not in self._parameters and name not in self._buffers:
                setattr(self, name, val)
            elif torch.is_tensor(val):
                constraint = self.constraint_for_parameter_name(name)
                if constraint is not None and constraint.enforced and not constraint.check_raw(val):
                    raise RuntimeError(
                        "Attempting to manually set a parameter value that is out of bounds of "
                        f"its current constraints, {constraint}. "
                        "Most likely, you want to do the following:\n likelihood = GaussianLikelihood"
                        "(noise_constraint=gpytorch.constraints.GreaterThan(better_lower_bound))"
                    )
                try:
                    self.__getattr__(name).data.copy_(val.expand_as(self.__getattr__(name)))
                except RuntimeError:
                    if not self._strict_init:
                        self.__getattr__(name).data = val
                    else:
                        self.__getattr__(name).data.copy_(val.view_as(self.__getattr__(name)))

            elif isinstance(val, float):
                constraint = self.constraint_for_parameter_name(name)
                if constraint is not None and not constraint.check_raw(val):
                    raise RuntimeError(
                        "Attempting to manually set a parameter value that is out of bounds of "
                        f"its current constraints, {constraint}. "
                        "Most likely, you want to do the following:\n likelihood = GaussianLikelihood"
                        "(noise_constraint=gpytorch.constraints.GreaterThan(better_lower_bound))"
                    )
                self.__getattr__(name).data.fill_(val)
            else:
                raise AttributeError("Type {t} not valid for initializing parameter {p}".format(t=type(val), p=name))

            # Ensure value is contained in support of prior (if present)
            prior_name = "_".join([name, "prior"])
            if prior_name in self._priors:
                prior, closure, _ = self._priors[prior_name]
                try:
                    prior._validate_sample(closure(self))
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
        from .variational._variational_distribution import _VariationalDistribution

        for module_prefix, module in self.named_modules():
            if not isinstance(module, _VariationalDistribution):
                for elem in module.named_parameters(prefix=module_prefix, recurse=False):
                    yield elem

    def named_priors(self, memo=None, prefix=""):
        """Returns an iterator over the module's priors, yielding the name of the prior,
        the prior, the associated parameter names, and the transformation callable.

        Yields:
            (string, Module, Prior, tuple((Parameter, callable)), callable): Tuple containing:
                - the name of the prior
                - the parent module of the prior
                - the prior
                - a tuple of tuples (param, transform), one for each of the parameters associated with the prior
                - the prior's transform to be called on the parameters
        """
        return _extract_named_priors(module=self, memo=None, prefix="")

    def named_constraints(self, memo=None, prefix=""):
        return _extract_named_constraints(module=self, memo=None, prefix="")

    def named_variational_parameters(self):
        from .variational._variational_distribution import _VariationalDistribution

        for module_prefix, module in self.named_modules():
            if isinstance(module, _VariationalDistribution):
                for elem in module.named_parameters(prefix=module_prefix, recurse=False):
                    yield elem

    def register_added_loss_term(self, name):
        self._added_loss_terms[name] = None

    def register_parameter(self, name, parameter):
        r"""
        Adds a parameter to the module. The parameter can be accessed as an attribute using the given name.

        Args:
            :attr:`name` (str):
                The name of the parameter
            :attr:`parameter` (torch.nn.Parameter):
                The parameter
        """
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
                the module instance and one or more parameters):
                single parameter without a transform: `.register_prior("foo_prior", foo_prior, "foo_param")`
                transform a single parameter (e.g. put a log-Normal prior on it):
                `.register_prior("foo_prior", NormalPrior(0, 1), lambda module: torch.log(module.foo_param))`
                function of multiple parameters:
                `.register_prior("foo2_prior", foo2_prior, lambda module: f(module.param1, module.param2)))`
            :attr:`setting_closure` (callable, optional):
                A function taking in the module instance and a tensor in (transformed) parameter space,
                initializing the internal parameter representation to the proper value by applying the
                inverse transform. Enables setting parametres directly in the transformed space, as well
                as sampling parameter values from priors (see `sample_from_prior`)

        """
        if isinstance(param_or_closure, str):
            if param_or_closure not in self._parameters and not hasattr(self, param_or_closure):
                raise AttributeError(
                    "Unknown parameter {name} for {module}".format(
                        name=param_or_closure, module=self.__class__.__name__
                    )
                    + " Make sure the parameter is registered before registering a prior."
                )

            def closure(module):
                return getattr(module, param_or_closure)

            if setting_closure is not None:
                raise RuntimeError("Must specify a closure instead of a parameter name when providing setting_closure")

            def setting_closure(module, val):
                return module.initialize(**{param_or_closure: val})

        else:
            if len(inspect.signature(param_or_closure).parameters) == 0:
                raise ValueError(
                    """As of version 1.4, `param_or_closure` must operate on a module instance. For example:

                    likelihood.noise_covar.register_prior(
                        "noise_std_prior",
                        gpytorch.priors.NormalPrior(0, 1),
                        lambda module: module.noise.sqrt()
                    )
                    """
                )
            if inspect.isfunction(setting_closure) and len(inspect.signature(setting_closure).parameters) < 2:
                raise ValueError(
                    """As of version 1.4, `setting_closure` must operate on a module instance and a tensor. For example:

                    kernel.register_prior(
                        "radius_prior",
                        gpytorch.priors.LogNormalPrior(0, 1),
                        lambda module: module.radius,
                        lambda module, value: m._set_radius(value),
                    )
                    """
                )
            closure = param_or_closure

        self.add_module(name, prior)
        self._priors[name] = (prior, closure, setting_closure)

    def register_constraint(self, param_name, constraint, replace=True):
        if param_name not in self._parameters:
            raise RuntimeError("Attempting to register constraint for nonexistent parameter.")

        constraint_name = param_name + "_constraint"
        if constraint_name in self._constraints:
            current_constraint = self._constraints[constraint_name]
        else:
            current_constraint = None

        if isinstance(current_constraint, Interval) and not replace:
            new_constraint = constraint.intersect(current_constraint)
        else:
            new_constraint = constraint

        self.add_module(constraint_name, new_constraint)
        self._constraints[constraint_name] = new_constraint

        # re-initialize the parameter if the constraint specifies an initial value
        if new_constraint.initial_value is not None:
            self.initialize(**{param_name: new_constraint.initial_value})

    def train(self, mode=True):
        # If we're going in training mode, we need to clear any pre-comptued caches from eval mode
        if (self.training and not mode) or mode:
            self._clear_cache()
        return super().train(mode=mode)

    def constraint_for_parameter_name(self, param_name):
        base_module = self
        base_name = param_name

        while "." in base_name:
            components = base_name.split(".")
            submodule_name = components[0]
            submodule = getattr(base_module, submodule_name)

            base_module = submodule
            base_name = ".".join(components[1:])

        try:
            constraint_name = base_name + "_constraint"
            return base_module._constraints.get(constraint_name)
        except AttributeError:  # submodule may not always be a gpytorch module
            return None

    def _load_state_hook_ignore_shapes(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if not self._load_strict_shapes:
            local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
            local_state = {k: v for k, v in local_name_params if v is not None}

            for name, param in local_state.items():
                key = prefix + name
                if key in state_dict:
                    param.data = state_dict[key].data

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # If we're loading from a state dict, we need to clear any precomputed caches
        self._clear_cache()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def load_strict_shapes(self, value):
        def apply_fn(module):
            module._load_strict_shapes = value

        self.apply(apply_fn)

    def named_parameters_and_constraints(self):
        for name, param in self.named_parameters():
            yield name, param, self.constraint_for_parameter_name(name)

    def sample_from_prior(self, prior_name):
        """Sample parameter values from prior. Modifies the module's parameters in-place."""
        if prior_name not in self._priors:
            raise RuntimeError("Unknown prior name '{}'".format(prior_name))
        prior, _, setting_closure = self._priors[prior_name]
        if setting_closure is None:
            raise RuntimeError("Must provide inverse transform to be able to sample from prior.")
        setting_closure(self, prior.sample())

    def pyro_sample_from_prior(self):
        """
        For each parameter in this Module and submodule that have defined priors, sample a value for that parameter
        from its corresponding prior with a pyro.sample primitive and load the resulting value in to the parameter.

        This method can be used in a Pyro model to conveniently define pyro sample sites for all
        parameters of the model that have GPyTorch priors registered to them.
        """
        return _pyro_sample_from_prior(module=self, memo=None, prefix="")

    def local_load_samples(self, samples_dict, memo, prefix):
        """
        Defines local behavior of this Module when loading parameters from a samples_dict generated by a Pyro
        sampling mechanism.

        The default behavior here should almost always be called from any overriding class. However, a class may
        want to add additional functionality, such as reshaping things to account for the fact that parameters will
        acquire an extra batch dimension corresponding to the number of samples drawn.
        """
        self._strict(False)
        for name, (prior, closure, setting_closure) in self._priors.items():
            if prior is not None and prior not in memo:
                memo.add(prior)
                setting_closure(self, samples_dict[prefix + ("." if prefix else "") + name])
        self._strict(True)

    def pyro_load_from_samples(self, samples_dict):
        """
        Convert this Module in to a batch Module by loading parameters from the given `samples_dict`. `samples_dict`
        is typically produced by a Pyro sampling mechanism.

        Note that the keys of the samples_dict should correspond to prior names (covar_module.outputscale_prior) rather
        than parameter names (covar_module.raw_outputscale), because we will use the setting_closure associated with
        the prior to properly set the unconstrained parameter.

        Args:
            :attr:`samples_dict` (dict): Dictionary mapping *prior names* to sample values.
        """
        return _pyro_load_from_samples(module=self, samples_dict=samples_dict, memo=None, prefix="")

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

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return super().__getattribute__(name)
            except AttributeError:
                raise e


def _validate_module_outputs(outputs):
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


def _set_strict(module, value, memo=None):
    if memo is None:
        memo = set()

    if hasattr(module, "_strict_init"):
        module._strict_init = value

    for mname, module_ in module.named_children():
        _set_strict(module_, value)


def _pyro_sample_from_prior(module, memo=None, prefix=""):
    try:
        import pyro
    except ImportError:
        raise RuntimeError("Cannot call pyro_sample_from_prior without pyro installed!")
    if memo is None:
        memo = set()
    if hasattr(module, "_priors"):
        for prior_name, (prior, closure, setting_closure) in module._priors.items():
            if prior is not None and prior not in memo:
                if setting_closure is None:
                    raise RuntimeError(
                        "Cannot use Pyro for sampling without a setting_closure for each prior,"
                        f" but the following prior had none: {prior_name}, {prior}."
                    )
                memo.add(prior)
                prior = prior.expand(closure(module).shape)
                value = pyro.sample(prefix + ("." if prefix else "") + prior_name, prior)
                setting_closure(module, value)

    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        _pyro_sample_from_prior(module=module_, memo=memo, prefix=submodule_prefix)


def _pyro_load_from_samples(module, samples_dict, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_priors"):
        module.local_load_samples(samples_dict, memo, prefix)

    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        _pyro_load_from_samples(module_, samples_dict, memo=memo, prefix=submodule_prefix)


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
                yield full_name, module, prior, closure, inv_closure
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, parent_module, prior, closure, inv_closure in _extract_named_priors(
            module_, memo=memo, prefix=submodule_prefix
        ):
            yield name, parent_module, prior, closure, inv_closure


def _extract_named_constraints(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_constraints"):
        for name, constraint in module._constraints.items():
            if constraint is not None and constraint not in memo:
                memo.add(constraint)
                full_name = ("." if prefix else "").join([prefix, name])
                yield full_name, constraint
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, constraint in _extract_named_constraints(module_, memo=memo, prefix=submodule_prefix):
            yield name, constraint
