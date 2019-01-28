#! /usr/bin/env python3

from abc import ABC, abstractproperty

import torch
from gpytorch.likelihoods import LikelihoodList
from gpytorch.models import GP
from torch.nn import ModuleList


class AbstractModelList(GP, ABC):
    @abstractproperty
    def num_outputs(self):
        """The model's number of outputs"""
        pass

    def forward_i(self, i, *args, **kwargs):
        """Forward restricted to the i-th output only"""
        raise NotImplementedError

    def likelihood_i(self, i, *args):
        """Evaluate likelihood of the i-th output only"""
        raise NotImplementedError


class IndependentModelList(AbstractModelList):
    def __init__(self, *models):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "IndependentModelList currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])

    @property
    def num_outputs(self):
        return len(self.models)

    def forward_i(self, i, *args, **kwargs):
        return self.models[i].forward(*args, **kwargs)

    def likelihood_i(self, i, *args):
        return self.likelihood.likelihoods[i](*args)

    def forward(self, *args, **kwargs):
        [model.forward(*args_, **kwargs) for model, args_ in zip(self.models, _get_tensor_args(*args))]

    def __call__(self, *args, **kwargs):
        return [model.__call__(*args_, **kwargs) for model, args_ in zip(self.models, _get_tensor_args(*args))]

    @property
    def train_inputs(self):
        return [model.train_inputs for model in self.models]

    @property
    def train_targets(self):
        return [model.train_targets for model in self.models]


def _get_tensor_args(*args):
    for arg in args:
        if torch.is_tensor(arg):
            yield (arg,)
        else:
            yield arg
