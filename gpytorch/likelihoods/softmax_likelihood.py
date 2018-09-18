from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from gpytorch import settings
from torch.distributions import Categorical

from ..distributions import MultivariateNormal
from .likelihood import Likelihood


class SoftmaxLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """

    def __init__(self, n_features, n_classes, mixing_weights_prior=None):
        super(SoftmaxLikelihood, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.register_parameter(
            name="mixing_weights",
            parameter=torch.nn.Parameter(torch.ones(n_classes, n_features).fill_(1. / n_features)),
            prior=mixing_weights_prior,
        )

    def forward(self, latent_func):
        """
        Computes predictive distributions p(y|x) given a latent distribution
        p(f|x). To do this, we solve the integral:

            p(y|x) = \int p(y|f)p(f|x) df

        Given that p(y=1|f) = \Phi(f), this integral is analytically tractable,
        and if \mu_f and \sigma^2_f are the mean and variance of p(f|x), the
        solution is given by:

            p(y|x) = \Phi(\frac{\mu}{\sqrt{1+\sigma^2_f}})
        """
        if not isinstance(latent_func, MultivariateNormal):
            raise RuntimeError(
                "SoftmaxLikelihood expects a multi-variate normally distributed latent function to make predictions"
            )

        n_samples = settings.num_likelihood_samples.value()
        samples = latent_func.sample(n_samples)
        samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
        if samples.ndimension() != 3:
            raise RuntimeError("f should have 3 dimensions: features x data x samples")
        n_features, n_data, _ = samples.size()
        if n_features != self.n_features:
            raise RuntimeError("There should be %d features" % self.n_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(n_features, n_samples * n_data))
        softmax = torch.nn.functional.softmax(mixed_fs.t()).view(n_data, n_samples, self.n_classes)
        return Categorical(probs=softmax.mean(1))

    def log_probability(self, latent_func, target):
        """
        Computes the log probability \sum_{i} \log \Phi(y_{i}f_{i}), where
        \Phi(y_{i}f_{i}) is computed by averaging over a set of s samples of
        f_{i} drawn from p(f|x).
        """
        n_samples = settings.num_likelihood_samples.value()
        samples = latent_func.sample(n_samples)
        samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
        if samples.ndimension() != 3:
            raise RuntimeError("f should have 3 dimensions: features x data x samples")
        n_features, n_data, _ = samples.size()
        if n_features != self.n_features:
            raise RuntimeError("There should be %d features" % self.n_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(n_features, n_samples * n_data))
        log_prob = -torch.nn.functional.cross_entropy(
            mixed_fs.t(), target.unsqueeze(1).repeat(1, n_samples).view(-1), size_average=False
        )
        return log_prob.div(n_samples)
