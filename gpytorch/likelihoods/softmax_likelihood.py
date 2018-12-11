#!/usr/bin/env python3

import torch
from torch.distributions import Categorical

from .. import settings
from ..distributions import MultivariateNormal
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

    def forward(self, latent_func):
        if not isinstance(latent_func, MultivariateNormal):
            raise RuntimeError(
                "SoftmaxLikelihood expects a multi-variate normally distributed latent function to make predictions"
            )

        n_samples = settings.num_likelihood_samples.value()
        samples = latent_func.rsample(sample_shape=torch.Size((n_samples,)))
        if samples.dim() == 2:
            samples = samples.unsqueeze(-1).transpose(-2, -1)
        samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
        if samples.ndimension() != 3:
            raise RuntimeError("f should have 3 dimensions: features x data x samples")
        num_features, n_data, _ = samples.size()
        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(num_features, n_samples * n_data))
        softmax = torch.nn.functional.softmax(mixed_fs.t(), 1).view(n_data, n_samples, self.n_classes)
        return Categorical(probs=softmax.mean(1))

    def variational_log_probability(self, latent_func, target):
        n_samples = settings.num_likelihood_samples.value()
        samples = latent_func.rsample(sample_shape=torch.Size((n_samples,)))
        if samples.dim() == 2:
            samples = samples.unsqueeze(-1).transpose(-2, -1)
        samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
        if samples.ndimension() != 3:
            raise RuntimeError("f should have 3 dimensions: features x data x samples")
        num_features, n_data, _ = samples.size()
        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(num_features, n_samples * n_data))
        log_prob = -torch.nn.functional.cross_entropy(
            mixed_fs.t(), target.unsqueeze(1).repeat(1, n_samples).view(-1), reduction="sum"
        )
        return log_prob.div(n_samples)
