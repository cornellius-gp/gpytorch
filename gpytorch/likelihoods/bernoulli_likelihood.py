from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.distributions import Bernoulli

from .. import settings
from ..distributions import MultivariateNormal
from ..functions import log_normal_cdf, normal_cdf
from .likelihood import Likelihood


class BernoulliLikelihood(Likelihood):
    """
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF \Phi(x)). Given the identity \Phi(-x) =
    1-\Phi(x), we can write the likelihood compactly as:

    p(Y=y|f)=\Phi(yf)
    """

    def forward(self, input):
        """
        Computes predictive distributions p(y|x) given a latent distribution
        p(f|x). To do this, we solve the integral:

            p(y|x) = \int p(y|f)p(f|x) df

        Given that p(y=1|f) = \Phi(f), this integral is analytically tractable,
        and if \mu_f and \sigma^2_f are the mean and variance of p(f|x), the
        solution is given by:

            p(y|x) = \Phi(\frac{\mu}{\sqrt{1+\sigma^2_f}})
        """
        if not isinstance(input, MultivariateNormal):
            raise RuntimeError(
                "BernoulliLikelihood expects a multi-variate normally distributed latent function to make predictions"
            )

        mean = input.mean
        var = input.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = normal_cdf(link)
        return Bernoulli(probs=output_probs)

    def variational_log_probability(self, latent_func, target):
        """
        Computes the log probability

            \sum_{i} \log \Phi(y_{i}f_{i}),

        where \Phi(y_{i}f_{i}) is computed by averaging over a set of s samples
        of f_{i} drawn from p(f|x).
        """
        num_samples = settings.num_likelihood_samples.value()
        samples = latent_func.rsample(torch.Size([num_samples])).view(-1)
        target = target.unsqueeze(0).repeat(num_samples, 1).view(-1)
        return log_normal_cdf(samples.mul(target)).sum().div(num_samples)
