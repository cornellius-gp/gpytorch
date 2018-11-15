from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import torch
from torch import nn
from torch.distributions import Distribution

from .lazy import LazyTensor

PRIOR_VALUE_WARNING = "Value of parameter {param} not valid for specified prior. Original exception:\n{exc}"


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._added_loss_terms = OrderedDict()
        self._priors = OrderedDict()
        self._parameter_transforms = OrderedDict()

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
            prior = self._priors.get(name)
            if prior is not None:
                param = self._parameters[name]
                tf = self._parameter_transforms.get(name)
                try:
                    prior._validate_sample(param if tf is None else tf(param))
                except ValueError as e:
                    raise ValueError(PRIOR_VALUE_WARNING.format(param=param, exc=e))
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

    def register_parameter(self, name, parameter, prior=None, transform=None):
        r"""
        Adds a parameter to the module. The parameter can be accessed as an attribute using the given name.

        Args:
            :attr:`name` (str):
                The name of the parameter
            :attr:`parameter` (torch.nn.Parameter):
                The parameter
            :attr:`prior` (Prior, optional):
                The prior for parameter. Can be accessed as an attribute using <name>_prior.
            :attr:`transform` (callable, optional):
                The transform to be used for this parameter. Typically used for parameterizing non-negative
                parameters such as kernel lengthscales, e.g. via softplus or exp transforms. The parameter
                can then be transformed by calling `.transform_parameter(name, parameter)` from the module.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError("Cannot assign parameter before Module.__init__() call")
        super(Module, self).register_parameter(name, parameter)
        if prior is not None:
            self.register_prior("_".join([name, "prior"]), prior, name)
        if transform is not None:
            self.register_parameter_transform(name, transform)

    def register_parameter_transform(self, parameter_name, transform):
        if parameter_name not in self._parameters:
            raise AttributeError(
                "Unknown parameter {name} for {module}".format(name=parameter_name, module=self.__class__.__name__)
            )
        self._parameter_transforms[parameter_name] = transform

    def register_prior(self, name, prior, *parameter_names, transform=None):
        """
        Adds a prior to the module. The prior can be accessed as an attribute using the given name.

        Args:
            :attr:`name` (str):
                The name of the prior
            :attr:`prior` (Prior):
                The prior to be registered`
            :attr:`parameter_names` (sequence of strings):
                The name(s) of the parameters (relative to the module) that are evaluated for this prior.
                The prior will be evaluated as `prior.log_prob(transform(*parameters))`, where transform=None corresponds
                to the identity transform, and parameters is the tuple of parameters corresponding to parameter_names
                (if applicable, the parameters are themselves transformed according to their associated parameter
                transformation registered in `register_parameter`). In the basic case of a single parameter without
                a transform, this is called as `.register_prior("foo_prior", foo_prior, "foo_param")`
            :attr:`transform` (callable, optional):
                The transform to be called on the specified parameters. Must take as many arguments as the number of
                parameters, each of them a tensor, and return itself a tensor. If ommitted, do not transform the parameter
                (only supported if registering a prior on a single parameter)

        .. note::

            Priors can operate on a transform of a single or of multiple parameters. This can be used, for instance,
            to put a prior over the ICM Kernel covariance matrix generated from covar_factor and log_var parameters.
        """
        if len(parameter_names) == 0:
            raise ValueError("Must pass at least one parameter name")
        elif len(parameter_names) > 1 and transform is None:
            raise ValueError("Must pass in a transform if specifying a prior operating on more than a single parameter")
        for parameter_name in parameter_names:
            if parameter_name not in self._parameters:
                raise AttributeError(
                    "Unknown parameter {name} for {module}".format(name=parameter_name, module=self.__class__.__name__)
                )
        self.add_module(name, prior)
        self._priors[name] = (prior, tuple(parameter_names), transform)

    def transform_parameter(self, parameter_name, parameter):
        transform = self._parameter_transforms.get(parameter_name)
        return parameter if transform is None else transform(parameter)

    def update_prior(self, name, prior, transform=None):
        """
        Update an existing prior with a new Prior object. The new prior operates on the same parameters
        as the original one under the new transform (if applicable).
        """
        if name not in self._priors:
            raise ValueError("Unknown prior - Please register the prior first using `register_prior`")
        parameter_names = self._priors[name][1]
        if len(parameter_names) > 1 and transform is None:
            raise ValueError(
                "Must specify a transform for priors defined over multiple parameters ({})".format(
                    ", ".join(parameter_names)
                )
            )
        self._priors[name] = (prior, parameter_names, transform)

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
        for name, (prior, pnames, tf) in module._priors.items():
            if prior is not None and prior not in memo:
                memo.add(prior)
                params_and_tfs = tuple(
                    (getattr(module, pname), module._parameter_transforms.get(pname)) for pname in pnames
                )
                full_name = ("." if prefix else "").join([prefix, name])
                yield full_name, prior, params_and_tfs, tf
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, prior, params_and_tfs, tf in _extract_named_priors(module_, memo=memo, prefix=submodule_prefix):
            yield name, prior, params_and_tfs, tf
