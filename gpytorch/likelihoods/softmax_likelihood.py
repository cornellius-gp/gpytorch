#!/usr/bin/env python3

import torch
from ..distributions import base_distributions
from .likelihood import Likelihood


class SoftmaxLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """

    def __init__(self, num_features, n_classes, mixing_weights_prior=None):
        super(SoftmaxLikelihood, self).__init__()
        self.num_features = num_features
        self.n_classes = n_classes
        self.register_parameter(
            name="mixing_weights",
            parameter=torch.nn.Parameter(torch.ones(n_classes, num_features).fill_(1.0 / num_features)),
        )
        if mixing_weights_prior is not None:
            self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights")

    def forward(self, function_samples, *params, **kwargs):
        num_features, num_data = function_samples.shape[-2:]
        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        mixed_fs = self.mixing_weights.matmul(function_samples)  # num_classes x num_data
        mixed_fs = mixed_fs.transpose(-1, -2)  # num_data x num_classes
        softmax = torch.nn.functional.softmax(mixed_fs, -1)
        return base_distributions.Categorical(probs=softmax)
