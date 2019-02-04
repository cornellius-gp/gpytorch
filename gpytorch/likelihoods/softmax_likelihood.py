#!/usr/bin/env python3

import torch
import warnings
from functools import partial
from torch.distributions import Categorical

from ..utils.quadrature import GaussHermiteQuadrature1D
from .likelihood import Likelihood


class SoftmaxLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """

    def __init__(
        self, num_features=None, n_classes=None, num_classes=None, mixing_weights=True, mixing_weights_prior=None
    ):
        if n_classes is not None:
            warnings.warn("n_classes keyword is deprecated. Use num_classes", DeprecationWarning)
            num_classes = n_classes
        elif num_classes is None:
            raise RuntimeError("Expected num_classes to be an integer. Got None")

        super(SoftmaxLikelihood, self).__init__()
        self.num_features = num_features if num_features is not None else num_classes
        self.num_classes = num_classes
        self.quadrature = GaussHermiteQuadrature1D()

        if mixing_weights:
            self.register_parameter(
                name="mixing_weights",
                parameter=torch.nn.Parameter(torch.ones(num_classes, num_features).fill_(1.0 / num_features)),
            )
            if mixing_weights_prior is not None:
                self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights")
        else:
            self.mixing_weights = None

    def _logit_samples(self, samples):
        # samples: num_classes (or num_features) x num_data x num_samples
        samples = samples.transpose(-3, -2)  # samples: num_data x num_classes (or num_features) x num_samples
        if self.mixing_weights is not None:
            _, num_features, _ = samples.shape
            if num_features != self.num_features:
                raise RuntimeError("There should be %d features" % self.num_features)
            samples = self.mixing_weights @ samples  # num_data x num_classes x num_samples

        return samples

    def _probs_function(self, samples):
        logit_samples = self._logit_samples(samples)  # ... x num_data x num_classes x num_samples
        return torch.nn.functional.softmax(logit_samples, dim=-2)

    def _log_likelihood_function(self, samples, targets):
        logit_samples = self._logit_samples(samples).transpose(-1, -2)  # ... x num_data x num_samples x num_classes
        log_likelihoods = Categorical(logits=logit_samples).log_prob(targets.unsqueeze(-1).expand(
            logit_samples.shape[:-1])
        )
        return log_likelihoods

    def forward(self, latent_func):
        return Categorical(probs=self.quadrature(self._probs_function, latent_func))

    def variational_log_probability(self, latent_func, targets):
        return self.quadrature(partial(self._log_likelihood_function, targets=targets), latent_func).sum()
