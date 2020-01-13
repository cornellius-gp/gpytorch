#!/usr/bin/env python3

import warnings

import torch

from ..distributions import Distribution, MultitaskMultivariateNormal, base_distributions
from .likelihood import Likelihood


class SoftmaxLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """

    def __init__(self, num_features=None, num_classes=None, mixing_weights=True, mixing_weights_prior=None, **kwargs):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes is required")
        self.num_classes = num_classes
        if mixing_weights:
            self.num_features = num_features
            if num_features is None:
                raise ValueError("num_features is required with mixing weights")
            self.register_parameter(
                name="mixing_weights",
                parameter=torch.nn.Parameter(torch.randn(num_classes, num_features).div_(num_features)),
            )
            if mixing_weights_prior is not None:
                self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights")
        else:
            self.num_features = num_classes
            self.mixing_weights = None

    def forward(self, function_samples, *params, **kwargs):
        num_data, num_features = function_samples.shape[-2:]

        # Catch legacy mode
        if num_data == self.num_features:
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated."
            )
            function_samples = function_samples.transpose(-1, -2)
            num_data, num_features = function_samples.shape[-2:]

        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        if self.mixing_weights is not None:
            mixed_fs = function_samples @ self.mixing_weights.t()  # num_classes x num_data
        else:
            mixed_fs = function_samples
        res = base_distributions.Categorical(logits=mixed_fs)
        return res

    def __call__(self, function, *params, **kwargs):
        if isinstance(function, Distribution) and not isinstance(function, MultitaskMultivariateNormal):
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated."
            )
            function = MultitaskMultivariateNormal.from_batch_mvn(function)
        return super().__call__(function, *params, **kwargs)
