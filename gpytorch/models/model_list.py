#! /usr/bin/env python3

from abc import ABC, abstractproperty

import torch
from torch.nn import ModuleList

from gpytorch.likelihoods import LikelihoodList
from gpytorch.models import GP


class AbstractModelList(GP, ABC):
    @abstractproperty
    def num_outputs(self):
        """The model's number of outputs"""
        pass

    def forward_i(self, i, *args, **kwargs):
        """Forward restricted to the i-th output only"""
        raise NotImplementedError

    def likelihood_i(self, i, *args, **kwargs):
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

    def likelihood_i(self, i, *args, **kwargs):
        return self.likelihood.likelihoods[i](*args, **kwargs)

    def forward(self, *args, **kwargs):
        return [model.forward(*args_, **kwargs) for model, args_ in zip(self.models, _get_tensor_args(*args))]

    def get_fantasy_model(self, inputs, targets, **kwargs):
        """
        Returns a new GP model that incorporates the specified inputs and targets as new training data.

        This is a simple wrapper that creates fantasy models for each of the models in the model list,
        and returns the same class of fantasy models.

        Args:
            - :attr:`inputs`: List of locations of fantasy observations, one for each model.
            - :attr:`targets` List of labels of fantasy observations, one for each model.

        Returns:
            - :class:`IndependentModelList`
                An `IndependentModelList` model, where each sub-model is the fantasy model of the respective
                sub-model in the original model at the corresponding input locations / labels.
        """

        if "noise" in kwargs:
            noise = kwargs.pop("noise")
            kwargs = [{**kwargs, "noise": noise_} if noise_ is not None else kwargs for noise_ in noise]
        else:
            kwargs = [kwargs] * len(inputs)

        fantasy_models = [
            model.get_fantasy_model(*inputs_, *targets_, **kwargs_)
            for model, inputs_, targets_, kwargs_ in zip(
                self.models, _get_tensor_args(*inputs), _get_tensor_args(*targets), kwargs
            )
        ]
        return self.__class__(*fantasy_models)

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
