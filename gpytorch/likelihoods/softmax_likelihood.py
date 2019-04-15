#!/usr/bin/env python3

import torch
from ..distributions import base_distributions
from .likelihood import Likelihood
from ..utils.deprecation import _deprecate_kwarg_with_transform


class SoftmaxLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """

    def __init__(
        self, num_features=None, num_classes=None, mixing_weights=True, mixing_weights_prior=None, **kwargs
    ):
        num_classes = _deprecate_kwarg_with_transform(
            kwargs, "n_classes", "num_classes", num_classes, lambda n: n
        )
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
        num_features, num_data = function_samples.shape[-2:]
        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        if self.mixing_weights is not None:
            mixed_fs = self.mixing_weights @ function_samples  # num_classes x num_data
        else:
            mixed_fs = function_samples
        mixed_fs = mixed_fs.transpose(-1, -2)  # num_data x num_classes
        res = base_distributions.Categorical(logits=mixed_fs)
        return res
