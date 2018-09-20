from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import torch
from ..module import Module
from ..functions import exact_predictive_mean, exact_predictive_covar
from ..distributions import MultivariateNormal, MultitaskMultivariateNormal
from ..likelihoods import GaussianLikelihood
from .. import settings


class ExactGP(Module):
    def __init__(self, train_inputs, train_targets, likelihood):
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        if train_inputs is not None and not all(torch.is_tensor(train_input) for train_input in train_inputs):
            raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")
        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError("ExactGP can only handle GaussianLikelihood")

        super(ExactGP, self).__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.mean_cache = None
        self.covar_cache = None

    def _apply(self, fn):
        if self.train_inputs is not None:
            self.train_inputs = tuple(fn(train_input) for train_input in self.train_inputs)
            self.train_targets = fn(self.train_targets)
        return super(ExactGP, self)._apply(fn)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """Set training data (does not re-fit model hyper-parameters)"""
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            for input, t_input in zip(inputs, self.train_inputs):
                for attr in {"shape", "dtype", "device"}:
                    if strict and getattr(input, attr) != getattr(t_input, attr):
                        raise RuntimeError("Cannot modify {attr} of inputs".format(attr=attr))
            self.train_inputs = inputs
        if targets is not None:
            for attr in {"shape", "dtype", "device"}:
                if strict and getattr(targets, attr) != getattr(self.train_targets, attr):
                    raise RuntimeError("Cannot modify {attr} of targets".format(attr=attr))
            self.train_targets = targets
        self.mean_cache = None
        self.covar_cache = None

    def train(self, mode=True):
        if mode:
            self.mean_cache = None
            self.covar_cache = None
        return super(ExactGP, self).train(mode)

    def __call__(self, *args, **kwargs):
        train_inputs = self.train_inputs
        inputs = tuple(i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args)

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            return super(ExactGP, self).__call__(*inputs, **kwargs)

        # Posterior mode
        else:
            if settings.debug.on() and self.train_inputs is not None:
                if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?", UserWarning
                    )

            # Exact inference
            if self.train_inputs is None:
                full_inputs = args
            else:
                full_inputs = tuple(
                    torch.cat([train_input, input], dim=-2) for train_input, input in zip(train_inputs, inputs)
                )
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.representation()

            n_tasks = 1
            if isinstance(full_output, MultitaskMultivariateNormal):
                n_tasks = full_output.n_tasks

            n_train = 0
            train_targets = None
            if self.train_targets is not None:
                if self.train_targets.ndimension() == 2 and n_tasks > 1:
                    # Multitask
                    full_mean = full_mean.view(-1)
                    n_train = self.train_targets.size(0)
                    train_targets = self.train_targets.view(-1)
                elif self.train_targets.ndimension() > 1:
                    # batch mode
                    full_mean = full_mean.view(full_mean.size(0), -1)
                    n_train = self.train_targets.size(1)
                    train_targets = self.train_targets.view(self.train_targets.size(0), -1)
                else:
                    n_train = self.train_targets.size(-1)
                    train_targets = self.train_targets

            predictive_mean, mean_cache = exact_predictive_mean(
                full_covar=full_covar,
                full_mean=full_mean,
                train_labels=train_targets,
                n_train=n_train,
                likelihood=self.likelihood,
                precomputed_cache=self.mean_cache,
            )
            predictive_covar, covar_cache = exact_predictive_covar(
                full_covar=full_covar, n_train=n_train, likelihood=self.likelihood, precomputed_cache=self.covar_cache
            )

            self.mean_cache = mean_cache
            self.covar_cache = covar_cache
            if n_tasks > 1:
                predictive_mean = predictive_mean.view(-1, n_tasks).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)
